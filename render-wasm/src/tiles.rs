use crate::render::Surfaces;
use crate::uuid::Uuid;
use crate::view::Viewbox;
use skia_safe as skia;
use std::collections::{HashMap, HashSet};
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct Tile(pub i32, pub i32);

impl Tile {
    pub fn from(x: i32, y: i32) -> Self {
        Tile(x, y)
    }

    #[inline(always)]
    pub fn x(&self) -> i32 {
        self.0
    }

    #[inline(always)]
    pub fn y(&self) -> i32 {
        self.1
    }

    #[inline(always)]
    pub fn get_rect_with_size(&self, tile_size: f32) -> skia::Rect {
        skia::Rect::from_xywh(
            self.0 as f32 * tile_size,
            self.1 as f32 * tile_size,
            tile_size,
            tile_size,
        )
    }

    /// Screen-space rect for this tile using the physical tile size (512×dpr).
    #[inline(always)]
    pub fn get_rect_with_offset(&self, offset: &skia::Point, screen_tile: f32) -> skia::Rect {
        skia::Rect::from_xywh(
            self.0 as f32 * screen_tile - offset.x,
            self.1 as f32 * screen_tile - offset.y,
            screen_tile,
            screen_tile,
        )
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
pub struct TileRect(pub i32, pub i32, pub i32, pub i32);

#[allow(dead_code)]
impl TileRect {
    pub fn empty() -> Self {
        Self(0, 0, 0, 0)
    }

    #[inline(always)]
    pub fn is_degenerate(&self) -> bool {
        self.left() > self.right() || self.top() > self.bottom()
    }

    #[inline(always)]
    pub fn len(&self) -> i32 {
        (self.width() + 1) * (self.height() + 1)
    }

    #[inline(always)]
    pub fn x1(&self) -> i32 {
        self.0
    }

    #[inline(always)]
    pub fn y1(&self) -> i32 {
        self.1
    }

    #[inline(always)]
    pub fn x2(&self) -> i32 {
        self.2
    }

    #[inline(always)]
    pub fn y2(&self) -> i32 {
        self.3
    }

    #[inline(always)]
    pub fn left(&self) -> i32 {
        self.0
    }

    #[inline(always)]
    pub fn top(&self) -> i32 {
        self.1
    }

    #[inline(always)]
    pub fn right(&self) -> i32 {
        self.2
    }

    #[inline(always)]
    pub fn bottom(&self) -> i32 {
        self.3
    }

    /// Inclusive tile count on X (matches `contains`: both `x1` and `x2` are included).
    #[inline(always)]
    pub fn columns(&self) -> i32 {
        self.x2() - self.x1() + 1
    }

    /// Inclusive tile count on Y (matches `contains`: both `y1` and `y2` are included).
    #[inline(always)]
    pub fn rows(&self) -> i32 {
        self.y2() - self.y1() + 1
    }

    #[inline(always)]
    pub fn width(&self) -> i32 {
        self.x2() - self.x1()
    }

    #[inline(always)]
    pub fn height(&self) -> i32 {
        self.y2() - self.y1()
    }

    #[inline(always)]
    pub fn contains(&self, tile: &Tile) -> bool {
        tile.x() >= self.left()
            && tile.y() >= self.top()
            && tile.x() <= self.right()
            && tile.y() <= self.bottom()
    }

    pub fn iter(self, inclusive: bool) -> TileRectIter {
        TileRectIter::new(self, inclusive)
    }
}

#[allow(dead_code)]
pub struct TileRectIter {
    rect: TileRect,
    inclusive: bool,
    index: i32,
    total: i32,
}

impl TileRectIter {
    fn new(rect: TileRect, inclusive: bool) -> Self {
        let width = rect.width() + if inclusive { 1 } else { 0 };
        let height = rect.height() + if inclusive { 1 } else { 0 };
        Self {
            rect,
            inclusive,
            index: 0,
            total: width * height,
        }
    }
}

impl Iterator for TileRectIter {
    type Item = Tile;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }

        let width = self.rect.width() + if self.inclusive { 1 } else { 0 };

        let x = self.rect.left() + self.index % width;
        let y = self.rect.top() + self.index / width;

        self.index += 1;

        Some(Tile::from(x, y))
    }
}

#[derive(Debug)]
pub struct TileViewbox {
    pub visible_rect: TileRect,
    pub interest_rect: TileRect,
    pub interest: i32,
    pub center: Tile,
}

impl TileViewbox {
    pub fn new_with_interest(viewbox: &Viewbox, interest: i32) -> Self {
        Self {
            visible_rect: get_tiles_for_viewbox(viewbox),
            interest_rect: get_tiles_for_viewbox_with_interest(viewbox, interest),
            interest,
            center: get_tile_center_for_viewbox(viewbox),
        }
    }

    pub fn update(&mut self, viewbox: &Viewbox) {
        self.visible_rect = get_tiles_for_viewbox(viewbox);
        self.interest_rect = get_tiles_for_viewbox_with_interest(viewbox, self.interest);
        self.center = get_tile_center_for_viewbox(viewbox);
    }

    pub fn set_interest(&mut self, interest: i32) {
        self.interest = interest;
    }

    pub fn is_visible(&self, tile: &Tile) -> bool {
        // TO CHECK self.interest_rect.contains(tile)
        self.visible_rect.contains(tile)
    }
}

pub const TILE_SIZE: f32 = 512.;

/// Max edge (px) for tile **paint** work surfaces (`Current`, shadows, …).
///
/// Without a cap, `512 × dpr` at DPR 4 yields 2048px tiles and 4096² effect
/// surfaces (~512 MB GPU for the scratchpads alone) and freezes on zoom.
pub const TILE_PAINT_SIZE_CAP: i32 = 1024;

/// Minimum atlas grid side so large paint tiles still pack enough slots
/// (e.g. 4096/8 → 512px slots → 64 entries).
pub const ATLAS_MIN_SLOTS_SIDE: i32 = 8;

// ---------------------------------------------------------------------------
// Three tile sizes (keep these distinct — mixing them caused HiDPI bugs):
//
//   paint_tile_size(dpr)  — raster into Current/effects (capped)
//   atlas_slot_size(...)  — packing cell in tile_atlas (≤ paint, capacity)
//   screen_tile_size(dpr) — placement on Target/Backbuffer (`512 × dpr`)
//
// Doc grid stays zoom-only: get_tile_size(zoom) = 512 / zoom.
// ---------------------------------------------------------------------------

/// Document-space size of one tile. Depends only on zoom (not DPR), so the
/// shape→tile grid stays stable across HiDPI.
#[inline(always)]
pub fn get_tile_size(zoom: f32) -> f32 {
    TILE_SIZE / zoom
}

/// GPU **paint** tile edge: `min(round(512 × dpr), TILE_PAINT_SIZE_CAP)`.
/// Sizes Current/effect surfaces and the paint CTM ([`tile_paint_scale`]).
#[inline(always)]
pub fn paint_tile_size(dpr: f32) -> i32 {
    let ideal = (TILE_SIZE * dpr).round().max(1.0) as i32;
    ideal.min(TILE_PAINT_SIZE_CAP)
}

/// Continuous **screen** size of one tile on Target/Backbuffer (`512 × dpr`,
/// uncapped). Atlas compose upscales from [`atlas_slot_size`] / paint when
/// the paint budget is below this.
#[inline(always)]
pub fn screen_tile_size(dpr: f32) -> f32 {
    TILE_SIZE * dpr
}

/// Integer screen tile edge for mosaic layout (cache surface, etc.).
#[inline(always)]
pub fn screen_tile_size_i32(dpr: f32) -> i32 {
    screen_tile_size(dpr).ceil().max(1.0) as i32
}

/// Atlas pack size: full paint tile when it fits, otherwise capped so the
/// atlas always has at least `ATLAS_MIN_SLOTS_SIDE²` slots.
#[inline(always)]
pub fn atlas_slot_size(paint_size: i32, atlas_texture_size: i32) -> i32 {
    let paint_size = paint_size.max(1);
    let atlas_texture_size = atlas_texture_size.max(1);
    let max_slot = (atlas_texture_size / ATLAS_MIN_SLOTS_SIDE).max(1);
    paint_size.min(max_slot)
}

/// CTM scale that maps a zoom-only doc tile onto an integer paint texture.
#[inline(always)]
pub fn tile_paint_scale(zoom: f32, paint_size: i32) -> f32 {
    paint_size as f32 * zoom / TILE_SIZE
}

#[inline(always)]
pub fn get_tile_dimensions(dpr: f32) -> skia::ISize {
    let s = paint_tile_size(dpr);
    (s, s).into()
}

pub fn get_tiles_for_rect(rect: skia::Rect, tile_size: f32) -> TileRect {
    // start
    let sx = (rect.left / tile_size).floor() as i32;
    let sy = (rect.top / tile_size).floor() as i32;
    // end
    let ex = (rect.right / tile_size).floor() as i32;
    let ey = (rect.bottom / tile_size).floor() as i32;
    TileRect(sx, sy, ex, ey)
}

pub fn get_tiles_for_viewbox(viewbox: &Viewbox) -> TileRect {
    let tile_size = get_tile_size(viewbox.zoom());
    get_tiles_for_rect(viewbox.area, tile_size)
}

pub fn get_tiles_for_viewbox_with_interest(viewbox: &Viewbox, interest: i32) -> TileRect {
    let TileRect(sx, sy, ex, ey) = get_tiles_for_viewbox(viewbox);
    TileRect(sx - interest, sy - interest, ex + interest, ey + interest)
}

pub fn get_tile_center_for_viewbox(viewbox: &Viewbox) -> Tile {
    let TileRect(sx, sy, ex, ey) = get_tiles_for_viewbox(viewbox);
    Tile((ex - sx) / 2, (ey - sy) / 2)
}

pub fn get_tile_pos(Tile(x, y): Tile, zoom: f32) -> (f32, f32) {
    let ts = get_tile_size(zoom);
    (x as f32 * ts, y as f32 * ts)
}

pub fn get_tile_rect(tile: Tile, zoom: f32) -> skia::Rect {
    let (tx, ty) = get_tile_pos(tile, zoom);
    let ts = get_tile_size(zoom);
    skia::Rect::from_xywh(tx, ty, ts, ts)
}

// This structure is useful to keep all the shape uuids by shape id.
pub struct TileHashMap {
    grid: HashMap<Tile, HashSet<Uuid>>,
    index: HashMap<Uuid, HashSet<Tile>>,
}

impl TileHashMap {
    pub fn new() -> Self {
        TileHashMap {
            grid: HashMap::new(),
            index: HashMap::new(),
        }
    }

    pub fn is_empty_at(&self, tile: Tile) -> bool {
        if let Some(uuids) = self.grid.get(&tile) {
            return uuids.is_empty();
        }
        true
    }

    pub fn get_shapes_at(&mut self, tile: Tile) -> Option<&HashSet<Uuid>> {
        self.grid.get(&tile)
    }

    pub fn remove_shape_at(&mut self, tile: Tile, id: Uuid) {
        if let Some(shapes) = self.grid.get_mut(&tile) {
            shapes.remove(&id);
        }

        if let Some(tiles) = self.index.get_mut(&id) {
            tiles.remove(&tile);
        }
    }

    pub fn get_tiles_of(&mut self, shape_id: Uuid) -> Option<&HashSet<Tile>> {
        self.index.get(&shape_id)
    }

    pub fn add_shape_at(&mut self, tile: Tile, shape_id: Uuid) {
        let tile_set = self.grid.entry(tile).or_default();
        tile_set.insert(shape_id);

        let index_set = self.index.entry(shape_id).or_default();
        index_set.insert(tile);
    }

    pub fn invalidate(&mut self) {
        self.grid.clear();
        self.index.clear();
    }
}

const VIEWPORT_DEFAULT_CAPACITY: usize = 24 * 12;

// This structure keeps the list of tiles that are in the pending list, the
// ones that are going to be rendered.
pub struct PendingTiles {
    pub list: Vec<Tile>,
    pub tile_order: Vec<(i32, Tile)>,
    pub tile_rect: TileRect,
    pub visible_cached: Vec<Tile>,
    pub visible_uncached: Vec<Tile>,
    pub interest_cached: Vec<Tile>,
    pub interest_uncached: Vec<Tile>,
}

impl PendingTiles {
    pub fn new() -> Self {
        Self {
            list: Vec::with_capacity(VIEWPORT_DEFAULT_CAPACITY),
            tile_order: Vec::with_capacity(VIEWPORT_DEFAULT_CAPACITY),
            tile_rect: TileRect::empty(),
            visible_cached: Vec::with_capacity(VIEWPORT_DEFAULT_CAPACITY),
            visible_uncached: Vec::with_capacity(VIEWPORT_DEFAULT_CAPACITY),
            interest_cached: Vec::with_capacity(VIEWPORT_DEFAULT_CAPACITY),
            interest_uncached: Vec::with_capacity(VIEWPORT_DEFAULT_CAPACITY),
        }
    }

    pub fn update(&mut self, tile_viewbox: &TileViewbox, surfaces: &Surfaces, only_visible: bool) {
        self.list.clear();

        // During interactive transform, skip the interest-area ring
        // entirely — the user is dragging, every rAF is on the critical
        // path, and pre-rendering tiles outside the viewport is wasted
        // work that just gets evicted on the next pointer move. The ring
        // is repopulated naturally on gesture end / on idle rAFs.
        let tile_rect = if only_visible {
            &tile_viewbox.visible_rect
        } else {
            &tile_viewbox.interest_rect
        };

        self.tile_rect = *tile_rect;

        // Partition tiles into 4 priority groups (highest priority = processed last due to pop()):
        // 1. visible + cached (fastest - just blit from cache)
        // 2. visible + uncached (user sees these, render next)
        // 3. interest + cached (pre-rendered area, blit from cache)
        // 4. interest + uncached (lowest priority - background pre-render)
        self.visible_cached.clear();
        self.visible_uncached.clear();
        self.interest_cached.clear();
        self.interest_uncached.clear();

        // Enumerate every tile in `tile_rect`, ordered by distance from the
        // rect center.
        let center_x = (tile_rect.x1() + tile_rect.x2()) / 2;
        let center_y = (tile_rect.y1() + tile_rect.y2()) / 2;

        self.tile_order.clear();

        for tile in tile_rect.iter(true) {
            let dx = tile.x() - center_x;
            let dy = tile.y() - center_y;
            self.tile_order.push((dx * dx + dy * dy, tile));
        }

        // Farthest first, since we use pop() to process the tiles
        // in order of priority (closest first)
        self.tile_order.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        for (_, tile) in self.tile_order.iter() {
            let tile = *tile;
            let is_visible = tile_viewbox.visible_rect.contains(&tile);
            let is_cached = surfaces.has_cached_tile_surface(tile);

            match (is_visible, is_cached) {
                (true, true) => self.visible_cached.push(tile),
                (true, false) => self.visible_uncached.push(tile),
                (false, true) => self.interest_cached.push(tile),
                (false, false) => self.interest_uncached.push(tile),
            }
        }

        self.list.extend(self.interest_uncached.iter());
        self.list.extend(self.interest_cached.iter());
        self.list.extend(self.visible_uncached.iter());
        self.list.extend(self.visible_cached.iter());
    }

    pub fn pop(&mut self) -> Option<Tile> {
        self.list.pop()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paint_tile_size_matches_ideal_until_cap() {
        assert_eq!(paint_tile_size(1.0), 512);
        assert_eq!(paint_tile_size(2.0), 1024);
        assert_eq!(paint_tile_size(2.0), TILE_PAINT_SIZE_CAP);
    }

    #[test]
    fn paint_tile_size_caps_high_dpr() {
        assert_eq!(paint_tile_size(3.0), TILE_PAINT_SIZE_CAP);
        assert_eq!(paint_tile_size(4.0), TILE_PAINT_SIZE_CAP);
        assert_eq!(paint_tile_size(4.0), 1024);
    }

    #[test]
    fn screen_tile_size_stays_uncapped() {
        assert_eq!(screen_tile_size(4.0), 2048.0);
        assert!(screen_tile_size(4.0) > paint_tile_size(4.0) as f32);
    }

    #[test]
    fn paint_scale_diverges_from_view_scale_when_capped() {
        let zoom = 1.0;
        let dpr = 4.0;
        let paint = tile_paint_scale(zoom, paint_tile_size(dpr));
        let view = zoom * dpr;
        assert_eq!(paint, 2.0);
        assert_eq!(view, 4.0);
        assert!(paint < view);
    }

    #[test]
    fn atlas_slot_preserves_paint_when_atlas_is_large_enough() {
        // 8192 / 8 = 1024 → paint 1024 fits at full res with 64 slots.
        assert_eq!(atlas_slot_size(1024, 8192), 1024);
        assert_eq!(atlas_slot_size(1024, 4096), 512);
        assert_eq!(atlas_slot_size(512, 4096), 512);
    }
}
