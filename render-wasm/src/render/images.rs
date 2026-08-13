use crate::math::Rect as MathRect;
use crate::shapes::ImageFill;
use crate::uuid::Uuid;

use crate::error::Result;
use crate::get_gpu_state;
use skia_safe::gpu::{surfaces, Budgeted, DirectContext};
use skia_safe::{self as skia, Codec, ISize, Size};
use std::cell::Cell;
use std::collections::HashMap;

pub type Image = skia::Image;

pub fn get_dest_rect(container: &MathRect, delta: f32) -> MathRect {
    MathRect::from_ltrb(
        container.left - delta,
        container.top - delta,
        container.right + delta,
        container.bottom + delta,
    )
}

pub fn get_source_rect(size: ISize, container: &MathRect, image_fill: &ImageFill) -> MathRect {
    let image_width = size.width as f32;
    let image_height = size.height as f32;

    // Container size
    let container_width = container.width();
    let container_height = container.height();

    let mut source_width = image_width;
    let mut source_height = image_height;
    let mut source_x = 0.;
    let mut source_y = 0.;

    let source_scale_y = image_height / container_height;
    let source_scale_x = image_width / container_width;

    if image_fill.keep_aspect_ratio() {
        // Calculate scale to ensure the image covers the container
        let image_aspect_ratio = image_width / image_height;
        let container_aspect_ratio = container_width / container_height;

        if image_aspect_ratio > container_aspect_ratio {
            // Image is taller, scale based on width to cover container
            source_width = container_width * source_scale_y;
            source_x = (image_width - source_width) / 2.0;
        } else {
            // Image is wider, scale based on height to cover container
            source_height = container_height * source_scale_x;
            source_y = (image_height - source_height) / 2.0;
        };
    }

    MathRect::from_xywh(source_x, source_y, source_width, source_height)
}

/// Longest design-space side of a shape, in device pixels at the given scale
/// (`zoom × dpr` for needed; `dpr` alone for the 100%-zoom display tier).
pub fn shape_side_px(selrect: &MathRect, scale: f32) -> i32 {
    let side = selrect.width().max(selrect.height()).max(1.0);
    (side * scale.max(1e-6)).ceil().max(1.0) as i32
}

fn rgba_bytes(image: &Image) -> usize {
    let d = image.dimensions();
    (d.width.max(0) as usize) * (d.height.max(0) as usize) * 4
}

fn fit_size(native: ISize, max_side: i32) -> ISize {
    let longest = native.width.max(native.height).max(1);
    if max_side <= 0 || longest <= max_side {
        return native;
    }
    let scale = max_side as f32 / longest as f32;
    ISize::new(
        (native.width as f32 * scale).round().max(1.0) as i32,
        (native.height as f32 * scale).round().max(1.0) as i32,
    )
}

/// Encoded raster kept in RAM, with optional GPU tiers:
/// - `display`: shape @ 100% zoom (eager, small)
/// - `full`: native resolution (lazy, for deep zoom)
struct RasterStored {
    raw: Vec<u8>,
    native: ISize,
    display: Option<Image>,
    display_side: i32,
    full: Option<Image>,
}

enum StoredImage {
    Raster(RasterStored),
    /// Legacy / thumbnail path: a single GPU texture (often from a shared GL tex).
    Gpu(Image),
    Svg {
        dom: skia::svg::Dom,
        size: Size,
        // Lazy raster for consumers that need a texture (stroke fills,
        // exports). The shape fill path draws the DOM directly instead.
        raster: Option<Image>,
    },
    /// Encoded bytes not yet classified (decode failed at add time).
    Raw(Vec<u8>),
}

struct StoredEntry {
    image: StoredImage,
    /// Approximate retained cost: encoded bytes + resident GPU RGBA.
    bytes: usize,
    /// LRU tick; `Cell` so read paths can touch it without `&mut self`.
    last_used: Cell<u64>,
}

pub struct ImageStore {
    images: HashMap<(Uuid, bool), StoredEntry>,
    total_bytes: usize,
    tick: Cell<u64>,
    /// gpu-only
    context: Option<Box<DirectContext>>,
}

/// Creates a Skia image from an existing WebGL texture.
/// This avoids re-decoding the image, as the browser has already decoded
/// and uploaded it to the GPU.
fn create_image_from_gl_texture(
    context: &mut Box<DirectContext>,
    texture_id: u32,
    width: i32,
    height: i32,
) -> Result<Image> {
    use skia_safe::gpu;
    use skia_safe::gpu::gl::TextureInfo;

    // Create a TextureInfo describing the existing GL texture
    let texture_info = TextureInfo {
        target: gl::TEXTURE_2D,
        id: texture_id,
        format: gl::RGBA8,
        protected: gpu::Protected::No,
    };

    // Create a backend texture from the GL texture using the new API
    let label = format!("shared_texture_{}", texture_id);
    let backend_texture = unsafe {
        gpu::backend_textures::make_gl((width, height), gpu::Mipmapped::No, texture_info, label)
    };

    // Create a Skia image from the backend texture
    // Use TopLeft origin because HTML images have their origin at top-left,
    // while WebGL textures traditionally use bottom-left
    let image = Image::from_texture(
        context.as_mut(),
        &backend_texture,
        gpu::SurfaceOrigin::TopLeft,
        skia::ColorType::RGBA8888,
        skia::AlphaType::Premul,
        None,
    )
    .ok_or(crate::error::Error::CriticalError(
        "Failed to create Skia image from GL texture".to_string(),
    ))?;

    Ok(image)
}

fn codec_native_size(raw_data: &[u8]) -> Option<ISize> {
    let data = unsafe { skia::Data::new_bytes(raw_data) };
    let codec = Codec::from_data(&data)?;
    let mut dimensions = codec.dimensions();
    if codec.origin().swaps_width_height() {
        dimensions.width = codec.dimensions().height;
        dimensions.height = codec.dimensions().width;
    }
    Some(dimensions)
}

/// Decode `raw_data` and upload a GPU texture whose longest side is at most
/// `max_side` (or native size when `max_side` is 0 / larger than native).
fn decode_image_to_max_side(
    context: &mut Box<DirectContext>,
    raw_data: &[u8],
    max_side: i32,
) -> Option<(Image, ISize)> {
    let data = unsafe { skia::Data::new_bytes(raw_data) };
    let codec = Codec::from_data(&data)?;
    let encoded = Image::from_encoded(&data)?;

    let mut native = codec.dimensions();
    if codec.origin().swaps_width_height() {
        native.width = codec.dimensions().height;
        native.height = codec.dimensions().width;
    }

    let dst = fit_size(native, max_side);
    let image_info = skia::ImageInfo::new_n32_premul(dst, None);

    let mut surface = surfaces::render_target(
        context,
        Budgeted::Yes,
        &image_info,
        None,
        None,
        None,
        true,
        false,
    )?;

    let dest_rect: MathRect =
        MathRect::from_xywh(0.0, 0.0, dst.width as f32, dst.height as f32);

    surface
        .canvas()
        .draw_image_rect(&encoded, None, dest_rect, &skia::Paint::default());

    Some((surface.image_snapshot(), native))
}

// Decode and upload to GPU at native size.
fn decode_image(context: &mut Box<DirectContext>, raw_data: &[u8]) -> Option<Image> {
    decode_image_to_max_side(context, raw_data, 0).map(|(img, _)| img)
}

fn raster_retained_bytes(raster: &RasterStored) -> usize {
    let mut bytes = raster.raw.len();
    if let Some(ref img) = raster.display {
        bytes += rgba_bytes(img);
    }
    if let Some(ref img) = raster.full {
        bytes += rgba_bytes(img);
    }
    bytes
}

// Size for SVGs without intrinsic dimensions nor a viewBox.
const DEFAULT_SVG_SIZE: f32 = 512.0;

// Parse an SVG and resolve its natural size. Skia codecs don't handle SVG,
// so this is the fallback when `decode_image` fails.
fn parse_svg(raw_data: &[u8]) -> Option<(skia::svg::Dom, Size)> {
    // An empty font manager: <text> elements inside SVG image fills won't
    // resolve typefaces. Wire the render state's font provider here if that
    // ever becomes a need.
    let font_mgr = skia::FontMgr::new();
    let mut dom = skia::svg::Dom::from_bytes(raw_data, font_mgr).ok()?;

    let mut size = dom.root().intrinsic_size();
    if size.is_empty() {
        // SVGs without width/height attributes have no intrinsic size;
        // fall back to the viewBox dimensions.
        size = dom
            .root()
            .view_box()
            .map(|vb| Size::new(vb.width(), vb.height()))
            .unwrap_or_else(|| Size::new(DEFAULT_SVG_SIZE, DEFAULT_SVG_SIZE));
    }

    // Ceil so the size matches the integer dimensions used when rasterizing.
    let size = Size::new(size.width.ceil(), size.height.ceil());
    if size.is_empty() {
        return None;
    }
    dom.set_container_size(size);

    Some((dom, size))
}

fn rasterize_svg(
    context: &mut Box<DirectContext>,
    dom: &skia::svg::Dom,
    size: Size,
) -> Option<Image> {
    let dimensions = ISize::new(size.width as i32, size.height as i32);
    let image_info = skia::ImageInfo::new_n32_premul(dimensions, None);
    let mut surface = surfaces::render_target(
        context,
        Budgeted::Yes,
        &image_info,
        None,
        None,
        None,
        true,
        false,
    )?;

    dom.render(surface.canvas());
    Some(surface.image_snapshot())
}

impl ImageStore {
    pub fn new() -> Self {
        let gpu_state = get_gpu_state();
        let context = &gpu_state.context;
        Self {
            images: HashMap::with_capacity(2048),
            total_bytes: 0,
            tick: Cell::new(0),
            context: Some(Box::new(context.clone())),
        }
    }

    /// GPU-free image store for the headless export path: no GPU context, so
    /// images are kept as encoded bytes and decoded on the CPU at draw time
    /// (see `get_cpu_image`).
    pub fn new_without_gpu() -> Self {
        Self {
            images: HashMap::with_capacity(16),
            total_bytes: 0,
            tick: Cell::new(0),
            context: None,
        }
    }

    /// Bumps the LRU clock and returns the new tick.
    fn next_tick(&self) -> u64 {
        let t = self.tick.get() + 1;
        self.tick.set(t);
        t
    }

    fn insert_entry(&mut self, key: (Uuid, bool), image: StoredImage, bytes: usize) {
        let last_used = Cell::new(self.next_tick());
        self.total_bytes += bytes;
        self.images.insert(
            key,
            StoredEntry {
                image,
                bytes,
                last_used,
            },
        );
    }

    fn recompute_entry_bytes(&mut self, key: (Uuid, bool)) {
        if let Some(entry) = self.images.get_mut(&key) {
            let new_bytes = match &entry.image {
                StoredImage::Raster(r) => raster_retained_bytes(r),
                StoredImage::Gpu(img) => rgba_bytes(img),
                StoredImage::Svg { .. } => entry.bytes, // keep prior encoded estimate
                StoredImage::Raw(raw) => raw.len(),
            };
            self.total_bytes = self
                .total_bytes
                .saturating_sub(entry.bytes)
                .saturating_add(new_bytes);
            entry.bytes = new_bytes;
        }
    }

    /// Evicts least-recently-used images until the store retains at most
    /// `max_bytes`. Meant to be called by the headless exporter *between*
    /// requests, so an image can never disappear under a running render;
    /// evicted images are simply re-provisioned by a later request that
    /// needs them (`is_image_cached` reports them as missing). Returns the
    /// number of evicted images.
    pub fn evict_to_budget(&mut self, max_bytes: usize) -> usize {
        if self.total_bytes <= max_bytes {
            return 0;
        }

        // Order the keys once instead of rescanning the whole store for the
        // minimum on every eviction.
        let mut keys: Vec<_> = self
            .images
            .iter()
            .map(|(key, entry)| (entry.last_used.get(), *key))
            .collect();
        keys.sort_unstable_by_key(|(last_used, _)| *last_used);

        let mut evicted = 0;
        for (_, key) in keys {
            if self.total_bytes <= max_bytes {
                break;
            }
            if let Some(entry) = self.images.remove(&key) {
                self.total_bytes -= entry.bytes;
                evicted += 1;
            }
        }
        evicted
    }

    /// Stores encoded image bytes. When `display_side` is set (shape @ 100% ×
    /// dpr), eagerly builds that GPU tier and keeps the full native decode
    /// lazy until deep zoom asks for it.
    pub fn add(
        &mut self,
        id: Uuid,
        is_thumbnail: bool,
        image_data: &[u8],
        display_side: Option<i32>,
    ) -> crate::error::Result<()> {
        let key = (id, is_thumbnail);

        if self.images.contains_key(&key) {
            if let Some(side) = display_side.filter(|s| *s > 0) {
                self.ensure_display_side(key, side);
            }
            return Ok(());
        }

        let raw_data = image_data.to_vec();
        let bytes = raw_data.len();

        match self.context.as_mut() {
            Some(_) => {
                if let Some((dom, size)) = parse_svg(&raw_data) {
                    self.insert_entry(
                        key,
                        StoredImage::Svg {
                            dom,
                            size,
                            raster: None,
                        },
                        bytes,
                    );
                } else if let Some(native) = codec_native_size(&raw_data) {
                    let mut raster = RasterStored {
                        raw: raw_data,
                        native,
                        display: None,
                        display_side: 0,
                        full: None,
                    };

                    // Thumbnails are already small: decode once at native.
                    // Otherwise eagerly build the display tier when we know
                    // the shape side at 100% zoom.
                    let eager_side = if is_thumbnail {
                        native.width.max(native.height)
                    } else {
                        display_side.unwrap_or(0).max(0)
                    };

                    if eager_side > 0 {
                        let target = fit_size(native, eager_side);
                        let target_side = target.width.max(target.height);
                        if let Some(context) = self.context.as_mut() {
                            if let Some((img, _)) =
                                decode_image_to_max_side(context, &raster.raw, target_side)
                            {
                                // Always park the eager decode in `display`.
                                // `full` is reserved for a larger lazy promote.
                                raster.display = Some(img);
                                raster.display_side =
                                    target_side.min(native.width.max(native.height));
                            }
                        }
                    }

                    let retained = raster_retained_bytes(&raster);
                    self.insert_entry(key, StoredImage::Raster(raster), retained);
                } else {
                    // The lazy re-decode in `get_internal` only retries raster codecs,
                    // so SVGs that fail to parse here stay raw.
                    self.insert_entry(key, StoredImage::Raw(raw_data), bytes);
                }
            }
            // GPU-free: keep the encoded bytes; decoded on the CPU at draw time.
            // SVGs still get parsed up front since that needs no GPU context.
            None => {
                if let Some((dom, size)) = parse_svg(&raw_data) {
                    self.insert_entry(
                        key,
                        StoredImage::Svg {
                            dom,
                            size,
                            raster: None,
                        },
                        bytes,
                    );
                } else {
                    self.insert_entry(key, StoredImage::Raw(raw_data), bytes);
                }
            }
        }
        Ok(())
    }

    /// Creates a Skia image from an existing WebGL texture, avoiding re-decoding.
    /// This is much more efficient as it reuses the texture that was already
    /// decoded and uploaded to GPU by the browser.
    pub fn add_image_from_gl_texture(
        &mut self,
        id: Uuid,
        is_thumbnail: bool,
        texture_id: u32,
        width: i32,
        height: i32,
    ) -> Result<()> {
        let key = (id, is_thumbnail);

        if self.images.contains_key(&key) {
            return Ok(());
        }

        // Create a Skia image from the existing GL texture
        let Some(context) = self.context.as_mut() else {
            return Err(crate::error::Error::CriticalError(
                "Cannot register a GL texture without a GPU context".to_string(),
            ));
        };
        let image = create_image_from_gl_texture(context, texture_id, width, height)?;
        let bytes = (width as usize) * (height as usize) * 4;
        self.insert_entry(key, StoredImage::Gpu(image), bytes);

        Ok(())
    }

    pub fn contains(&self, id: &Uuid, is_thumbnail: bool) -> bool {
        self.images.contains_key(&(*id, is_thumbnail))
    }

    /// Grow/create the display GPU tier so it covers at least `side` px
    /// (capped to native). Used when the shape grows while zoomed out.
    fn ensure_display_side(&mut self, key: (Uuid, bool), side: i32) {
        let Some(entry) = self.images.get_mut(&key) else {
            return;
        };
        let StoredImage::Raster(raster) = &mut entry.image else {
            return;
        };
        let native_side = raster.native.width.max(raster.native.height);
        let target = side.min(native_side).max(1);
        if raster.display_side >= target {
            return;
        }
        // If we already have full and target is native, display is redundant.
        if target >= native_side {
            if raster.full.is_none() && raster.display.is_none() {
                let raw = raster.raw.clone();
                let Some(context) = self.context.as_mut() else {
                    return;
                };
                if let Some((img, _)) = decode_image_to_max_side(context, &raw, 0) {
                    if let Some(entry) = self.images.get_mut(&key) {
                        if let StoredImage::Raster(raster) = &mut entry.image {
                            // Native-sized tier lives in `display` so eviction
                            // logic never drops the only GPU image.
                            raster.display = Some(img);
                            raster.display_side = native_side;
                            raster.full = None;
                        }
                    }
                    self.recompute_entry_bytes(key);
                }
            } else if let Some(entry) = self.images.get_mut(&key) {
                if let StoredImage::Raster(raster) = &mut entry.image {
                    if raster.display.is_none() {
                        if let Some(full) = raster.full.take() {
                            raster.display = Some(full);
                        }
                    }
                    raster.display_side = native_side;
                    raster.full = None;
                }
                self.recompute_entry_bytes(key);
            }
            return;
        }

        let raw = match self.images.get(&key) {
            Some(StoredEntry {
                image: StoredImage::Raster(r),
                ..
            }) => r.raw.clone(),
            _ => return,
        };
        let Some(context) = self.context.as_mut() else {
            return;
        };
        let Some((img, _)) = decode_image_to_max_side(context, &raw, target) else {
            return;
        };
        if let Some(entry) = self.images.get_mut(&key) {
            if let StoredImage::Raster(raster) = &mut entry.image {
                raster.display = Some(img);
                raster.display_side = target;
            }
        }
        self.recompute_entry_bytes(key);
    }

    fn ensure_full(&mut self, key: (Uuid, bool)) -> bool {
        let raw = match self.images.get(&key) {
            Some(StoredEntry {
                image: StoredImage::Raster(r),
                ..
            }) => {
                if r.full.is_some() {
                    return true;
                }
                r.raw.clone()
            }
            _ => return false,
        };
        let Some(context) = self.context.as_mut() else {
            return false;
        };
        let Some((img, _)) = decode_image_to_max_side(context, &raw, 0) else {
            return false;
        };
        if let Some(entry) = self.images.get_mut(&key) {
            if let StoredImage::Raster(raster) = &mut entry.image {
                raster.full = Some(img);
            }
        }
        self.recompute_entry_bytes(key);
        true
    }

    fn drop_full_if_unneeded(&mut self, key: (Uuid, bool), needed_side: i32) {
        let should_drop = {
            let Some(entry) = self.images.get(&key) else {
                return;
            };
            let StoredImage::Raster(raster) = &entry.image else {
                return;
            };
            if raster.full.is_none() {
                return;
            }
            // Only evict full when a real display texture still covers the
            // request. If display was cleared because the tier is native-sized
            // (full is the only GPU image), dropping it leaves nothing to draw.
            raster.display.is_some()
                && raster.display_side > 0
                && needed_side <= raster.display_side
        };
        if !should_drop {
            return;
        }
        if let Some(entry) = self.images.get_mut(&key) {
            if let StoredImage::Raster(raster) = &mut entry.image {
                raster.full = None;
            }
        }
        self.recompute_entry_bytes(key);
    }

    /// Picks display vs full for painting. `display_side` is shape@100%×dpr;
    /// `needed_side` is shape×view_scale (zoom×dpr).
    pub fn get_for_draw(
        &mut self,
        id: &Uuid,
        display_side: i32,
        needed_side: i32,
    ) -> Option<&Image> {
        let key = if self.images.contains_key(&(*id, false)) {
            (*id, false)
        } else if self.images.contains_key(&(*id, true)) {
            (*id, true)
        } else {
            return None;
        };

        // Promote Raw → Raster on first draw if needed.
        if matches!(
            self.images.get(&key).map(|e| &e.image),
            Some(StoredImage::Raw(_))
        ) {
            let raw = match self.images.remove(&key) {
                Some(StoredEntry {
                    image: StoredImage::Raw(raw),
                    bytes,
                    last_used,
                }) => {
                    self.total_bytes = self.total_bytes.saturating_sub(bytes);
                    let _ = last_used;
                    raw
                }
                Some(other) => {
                    self.images.insert(key, other);
                    return None;
                }
                None => return None,
            };
            if let Some(native) = codec_native_size(&raw) {
                let retained = raw.len();
                self.insert_entry(
                    key,
                    StoredImage::Raster(RasterStored {
                        raw,
                        native,
                        display: None,
                        display_side: 0,
                        full: None,
                    }),
                    retained,
                );
            } else {
                self.insert_entry(key, StoredImage::Raw(raw), 0);
                return None;
            }
        }

        if display_side > 0 {
            self.ensure_display_side(key, display_side);
        }

        let use_full = {
            let entry = self.images.get(&key)?;
            match &entry.image {
                StoredImage::Raster(r) => {
                    // Need full when display is missing or too small for needed.
                    let display_covers = r.display.is_some()
                        && r.display_side > 0
                        && needed_side <= r.display_side;
                    !display_covers
                }
                _ => false,
            }
        };

        if use_full {
            self.ensure_full(key);
        } else {
            self.drop_full_if_unneeded(key, needed_side);
        }

        if matches!(
            self.images.get(&key).map(|e| &e.image),
            Some(StoredImage::Svg { .. })
        ) {
            return self.get_internal(id, key.1);
        }

        let tick = self.next_tick();
        let entry = self.images.get(&key)?;
        entry.last_used.set(tick);
        match &entry.image {
            StoredImage::Raster(r) => {
                let prefer_display = r.display.is_some()
                    && r.display_side > 0
                    && needed_side <= r.display_side;
                if prefer_display {
                    r.display.as_ref()
                } else {
                    // Prefer full when needed; fall back to display if full
                    // decode failed (e.g. GPU limit) so the shape stays visible.
                    r.full.as_ref().or(r.display.as_ref())
                }
            }
            StoredImage::Gpu(img) => Some(img),
            StoredImage::Svg { .. } => None, // handled above
            StoredImage::Raw(_) => None,
        }
    }

    pub fn get(&mut self, id: &Uuid) -> Option<&Image> {
        // Legacy callers without LOD: prefer full when present, else display.
        let has_full = self.images.contains_key(&(*id, false));
        let key_thumb = !has_full;
        if has_full {
            // Request a huge needed side so full is materialized when possible.
            self.get_for_draw(id, 0, i32::MAX)
        } else {
            self.get_internal(id, key_thumb)
        }
    }

    pub fn get_cpu_image(&mut self, id: &Uuid) -> Option<Image> {
        // GPU path: promote to a texture, then copy to a CPU image.
        if self.context.is_some() {
            let gpu_image = self.get(id)?.clone();
            let context = self.context.as_mut()?;
            return gpu_image.make_non_texture_image(context.as_mut());
        }
        // Headless (no GPU context): decode the stored encoded bytes directly to
        // a CPU image, which draws fine on a raster/PDF canvas. Try full first,
        // then thumbnail.
        self.decode_raw_cpu_image(id, false)
            .or_else(|| self.decode_raw_cpu_image(id, true))
    }

    fn decode_raw_cpu_image(&self, id: &Uuid, is_thumbnail: bool) -> Option<Image> {
        let entry = self.images.get(&(*id, is_thumbnail))?;
        entry.last_used.set(self.next_tick());
        match &entry.image {
            StoredImage::Raw(raw_data) => {
                let data = unsafe { skia::Data::new_bytes(raw_data) };
                Image::from_encoded(&data)
            }
            StoredImage::Raster(r) => {
                let data = unsafe { skia::Data::new_bytes(&r.raw) };
                Image::from_encoded(&data)
            }
            StoredImage::Gpu(img) => Some(img.clone()),
            StoredImage::Svg { dom, size, .. } => {
                // No GPU context in the headless path: rasterize on a CPU
                // surface instead of `rasterize_svg` (which needs one).
                let dimensions = ISize::new(size.width as i32, size.height as i32);
                let mut surface = skia::surfaces::raster_n32_premul(dimensions)?;
                dom.render(surface.canvas());
                Some(surface.image_snapshot())
            }
        }
    }

    /// Vector access for SVG images: the fill render path draws the DOM
    /// directly so it stays crisp at any zoom level.
    pub fn get_svg(&self, id: &Uuid) -> Option<(&skia::svg::Dom, Size)> {
        let entry = self
            .images
            .get(&(*id, false))
            .or_else(|| self.images.get(&(*id, true)))?;
        entry.last_used.set(self.next_tick());
        match &entry.image {
            StoredImage::Svg { dom, size, .. } => Some((dom, *size)),
            _ => None,
        }
    }

    fn get_internal(&mut self, id: &Uuid, is_thumbnail: bool) -> Option<&Image> {
        let key = (*id, is_thumbnail);
        let tick = self.tick.get() + 1;
        self.tick.set(tick);

        let needs_full_decode = matches!(
            self.images.get(&key).map(|e| &e.image),
            Some(StoredImage::Raster(r)) if r.full.is_none() && r.display.is_none()
        );
        let needs_gpu_from_raw = matches!(
            self.images.get(&key).map(|e| &e.image),
            Some(StoredImage::Raw(_))
        );
        let needs_svg_raster = matches!(
            self.images.get(&key).map(|e| &e.image),
            Some(StoredImage::Svg { raster: None, .. })
        );

        if needs_full_decode {
            let raw = match self.images.get(&key) {
                Some(StoredEntry {
                    image: StoredImage::Raster(r),
                    ..
                }) => r.raw.clone(),
                _ => return None,
            };
            let context = self.context.as_mut()?;
            let gpu_image = decode_image(context, &raw)?;
            if let Some(entry) = self.images.get_mut(&key) {
                if let StoredImage::Raster(r) = &mut entry.image {
                    r.full = Some(gpu_image);
                }
            }
            self.recompute_entry_bytes(key);
        } else if needs_gpu_from_raw {
            let raw = match self.images.get(&key) {
                Some(StoredEntry {
                    image: StoredImage::Raw(raw),
                    ..
                }) => raw.clone(),
                _ => return None,
            };
            let context = self.context.as_mut()?;
            let gpu_image = decode_image(context, &raw)?;
            if let Some(entry) = self.images.get_mut(&key) {
                entry.image = StoredImage::Gpu(gpu_image);
            }
        } else if needs_svg_raster {
            let ImageStore {
                context, images, ..
            } = self;
            let context = context.as_mut()?;
            let entry = images.get_mut(&key)?;
            if let StoredImage::Svg { dom, size, raster } = &mut entry.image {
                *raster = rasterize_svg(context, dom, *size);
            }
        }

        let entry = self.images.get(&key)?;
        entry.last_used.set(tick);
        match &entry.image {
            StoredImage::Gpu(img) => Some(img),
            StoredImage::Raster(r) => r.full.as_ref().or(r.display.as_ref()),
            StoredImage::Svg { raster, .. } => raster.as_ref(),
            StoredImage::Raw(_) => None,
        }
    }
}
