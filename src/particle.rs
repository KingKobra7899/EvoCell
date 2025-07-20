use nalgebra::Vector2;
use std::ops::AddAssign;
use std::f32::consts::PI;
use raqote::{DrawTarget, PathBuilder, Source, SolidSource, DrawOptions};

pub struct Particle {
    pub pos: Vector2<f32>,
    pub old_pos: Vector2<f32>,
    pub acc: Vector2<f32>,
    pub mass: f32,
    pub rad: f32,
}

impl Particle {
    pub fn new(def_pos: Vector2<f32>, def_mass: f32, def_rad: f32) -> Self {
        Particle {
            pos: def_pos,
            old_pos: def_pos,
            acc: Vector2::new(0.0, 0.0),
            mass: def_mass,
            rad: def_rad,
        }
    }

    pub fn render(&self, dt: &mut DrawTarget) {
        let mut pb = PathBuilder::new();
        pb.arc(self.pos.x, self.pos.y, self.rad, 0.0, 2.0 * PI);
        let path = pb.finish();

        dt.fill(
            &path,
            &Source::Solid(SolidSource::from_unpremultiplied_argb(0xff, 0xff, 0x80, 0x80)),
            &DrawOptions::new(),
        );
    }

    pub fn accelerate(&mut self, force: Vector2<f32>) {
        self.acc += force / self.mass;
    }

    pub fn integrate_pos(&mut self, delta_time: f32, grav: Vector2<f32>) {
        self.accelerate(grav * self.mass);
        let new_pos = (2.0 * self.pos) - self.old_pos + self.acc * delta_time * delta_time;
        self.old_pos = self.pos;
        self.pos = new_pos;
        self.acc = Vector2::new(0.0, 0.0);
    }
}
