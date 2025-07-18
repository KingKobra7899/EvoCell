use minifb::{MouseMode, Window, WindowOptions, ScaleMode, Scale};
use raqote::{DrawTarget, SolidSource, Source, DrawOptions, PathBuilder, Point, Transform, StrokeStyle};
use font_kit::family_name::FamilyName;
use font_kit::properties::Properties;
use font_kit::source::SystemSource;

fn main() {
    const WIDTH: usize = 500;
    const HEIGHT: usize = 500;
    let mut window = Window::new("window", WIDTH, HEIGHT, WindowOptions::default()).unwrap();

    let size = window.get_size();
    let mut dt = DrawTarget::new(size.0 as i32, size.1 as i32);

    let font = SystemSource::new()
        .select_best_match(&[FamilyName::SansSerif], &Properties::new())
        .unwrap()
        .load()
        .unwrap();

    loop {
        dt.clear(SolidSource::from_unpremultiplied_argb(0xff, 0xff, 0xff, 0xff));
        let mut pb = PathBuilder::new();
        if let Some(pos) = window.get_mouse_pos(MouseMode::Clamp) {

            pb.rect(pos.0, pos.1, 100., 130.);
            let path = pb.finish();
            dt.fill(&path, &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0, 0xff, 0)), &DrawOptions::new());

            let pos_string = format!("{:?}", pos);
            dt.draw_text(&font, 36., &pos_string, Point::new(0., 100.),
                        &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0, 0, 0)),
                        &DrawOptions::new(),
            );

            window.update_with_buffer(dt.get_data(), size.0, size.1).unwrap();
        }
    }
}