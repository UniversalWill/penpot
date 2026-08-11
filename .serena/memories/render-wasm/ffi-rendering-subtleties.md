# render-wasm FFI and Rendering Subtleties

## FFI state and errors

- The renderer uses one unsafe global `STATE`; the `with_state*` macros currently panic on invalid state pointer. Treat state pointer validity as critical, not recoverable.
- `#[wasm_error]` clears the error code on entry. Recoverable errors set code `0x01`, critical errors/panics set `0x02`, free the byte buffer, then panic so the CLJS bridge can catch and inspect `_read_error_code`.
- The frontend bridge maps `0x01` to `:non-blocking` and `0x02` to `:panic` in ex-data (`:type :wasm-error`). Check actual bridge code if changing names; older comments/docs may use different labels.
- WASM byte transfer is a single global slot. A caller that receives a pointer result must read and free it before another byte payload is written; errors free the slot via `#[wasm_error]`.

## Shape pool and loading

- Shapes are UUID-indexed, and hierarchy/structure is tracked separately. `ShapesPool::get` may return a cached modified clone when modifiers, structure, scale-content, or bool handling apply; `get_raw` bypasses those derived values.
- Bulk loading uses a `loading` flag. `touch_current` / `touch_shape` avoid tile invalidation while loading; text layouts and final view setup must happen after loading ends.
- Many setters mutate only the current shape selected by `use_shape` / current-shape APIs. If no current shape is selected, some mutation blocks are skipped silently.
- `set_parent_for_current_shape` only sets parent metadata and invalidates parent geometry; children must be updated separately to avoid duplicate children.
- Child deletion marks descendants deleted and removes them from all indexed tiles, preserving undo/redo while avoiding stale pixels after panning.

## Tile/render behavior

- Three device sizes (do not conflate): `paint_tile_size(dpr)` (raster/Current, capped),
  `atlas_slot_size(paint, atlas)` (tile_atlas packing), `screen_tile_size(dpr)` (Target/
  Backbuffer placement, uncapped `512×dpr`). Doc grid is zoom-only: `512/zoom`.
- Surfaces allocate with `effective_paint_tile_size` = `min(paint, atlas_slot max)` so
  DPR 2 on a 4096² atlas paints 512 (not 1024→downscale-to-512). Overpainting the atlas
  was pure GPU waste on zoom settle.
- Raster `Fill::Image`: skip `save_layer` unless the shape has an image filter; plain
  Rect/Frame (no corners) also skip the container clip (`draw_image_fill` in fills.rs).
- Scales: `get_paint_scale()` matches tile CTM; `get_view_scale()` is `zoom×dpr` for
  viewport/backbuffer mapping. `get_scale()` is an alias of paint scale (legacy name).
- Zoom settle: visible tiles present via `FrameType::ViewportReady` before interest-ring
  work; crop-cache rebuild is deferred to the later `Full` so the soft→sharp snap is
  compose+present only.
- Interactive transforms are distinct from viewport fast mode. `set_modifiers_start` enables fast mode and interactive transform; interactive transform still flushes each animation frame.
- During interactive transform, modifier tile invalidation is deferred to `render()` once per rAF. Outside interactive transform, `set_modifiers` rebuilds modifier tiles immediately.
- `set_modifiers_end` disables fast/interactive state and cancels pending async render; the caller must request the final full-quality render.
- Plain viewport fast mode (`options.is_viewport_interaction()`) renders from cache and does not flush target output inside `process_animation_frame`; interactive transforms do flush.
- Zoom changes rebuild the tile index while preserving cached tile textures. Avoid replacing that path with shallow rebuilds if blur/shadow cache preservation matters.
- Pending tile priority is intentionally reversed by pop order; check the queue construction before changing tile scheduling.