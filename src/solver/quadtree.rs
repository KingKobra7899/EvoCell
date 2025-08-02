use nalgebra::Vector2;

#[derive(Clone, Copy)]
pub struct Rect{
    pub(crate) x:f32, //centered at (x, y), w distance from center to edge, h distance from center to edge
    pub(crate) y:f32,
    pub(crate) w:f32,
    pub(crate) h:f32
}

impl Rect {
    pub fn new(x: f32, y: f32, w: f32, h: f32) -> Rect {
        return Rect { x, y, w, h };
    }

    pub fn point_is_in(&self, point: Vector2<f32>) -> bool {
        
        point.x >= self.x - self.w && 
        point.x <= self.x + self.w && 
        point.y >= self.y - self.h && 
        point.y <= self.y + self.h
    }

    pub fn intersects(&self, other: Rect) -> bool {
      
        let x_overlap = (self.x + self.w) >= (other.x - other.w) && 
                       (other.x + other.w) >= (self.x - self.w);
        
        
        let y_overlap = (self.y + self.h) >= (other.y - other.h) && 
                       (other.y + other.h) >= (self.y - self.h);
        
        
        x_overlap && y_overlap
    }

    pub fn intersects_cone(
        &self,
        cone_origin: Vector2<f32>,
        direction: Vector2<f32>,
        radius: f32,
        angle: f32,
    ) -> bool {
        // Normalize the direction vector
        let dir_norm = direction.normalize();

        // Step 1: Check if cone origin is inside the rectangle
        if self.point_is_in(cone_origin) {
            return true;
        }

        // Step 2: Check if any rectangle corner is inside the cone
        let corners = [
            Vector2::new(self.x - self.w, self.y - self.h),
            Vector2::new(self.x + self.w, self.y - self.h),
            Vector2::new(self.x + self.w, self.y + self.h),
            Vector2::new(self.x - self.w, self.y + self.h),
        ];

        for &corner in &corners {
            if Self::point_in_cone(corner, cone_origin, dir_norm, radius, angle) {
                return true;
            }
        }

        // Step 3: Check if any point along the cone arc is inside the rectangle
        let num_samples = 20;
        for i in 0..=num_samples {
            let a = -angle + (2.0 * angle * i as f32) / num_samples as f32;
            let dir = rotate_vector(dir_norm, a);
            let point_on_arc = cone_origin + dir * radius;
            if self.point_is_in(point_on_arc) {
                return true;
            }
        }

        
        false
    }

    fn point_in_cone(
        point: Vector2<f32>,
        origin: Vector2<f32>,
        direction: Vector2<f32>,
        radius: f32,
        angle: f32,
    ) -> bool {
        let to_point = point - origin;
        let distance = to_point.norm();

        if distance > radius {
            return false;
        }

        let to_point_normalized = to_point.normalize();
        let dot = direction.dot(&to_point_normalized);
        let theta = dot.acos(); 

        theta <= angle
    }
}


fn rotate_vector(v: Vector2<f32>, angle: f32) -> Vector2<f32> {
    let cos_a = angle.cos();
    let sin_a = angle.sin();
    Vector2::new(
        v.x * cos_a - v.y * sin_a,
        v.x * sin_a + v.y * cos_a,
    )
}

pub struct QuadTree{
    boundary: Rect,
    capacity: i32,
    indices: Vec<i32>,
    points: Vec<Vector2<f32>>,
    northeast: Option<Box<QuadTree>>,
    northwest: Option<Box<QuadTree>>,
    southeast: Option<Box<QuadTree>>,
    southwest: Option<Box<QuadTree>>,

    divided: bool
}

impl QuadTree{
    pub fn new(boundary: Rect, capacity: i32) -> Self {
        QuadTree {
            boundary: boundary,
            capacity: capacity,
            indices: Vec::new(),
            points: Vec::new(),
            northeast: None,
            northwest: None,
            southeast: None,
            southwest: None,
            divided: false
        }
    }

    pub fn insert(&mut self, point: &Vector2<f32>, index: i32) {
       
        if !self.boundary.point_is_in(*point) {
            return;
        }
    
        
        if self.indices.len() < self.capacity as usize {
            self.indices.push(index);
            self.points.push(*point);
        } else {
            
            if !self.divided {
                self.subdivide();
            }
    
            let x = point.x;
            let y = point.y;
            let center_x = self.boundary.x;
            let center_y = self.boundary.y;
    
            if x >= center_x {
                if y >= center_y {
                    self.southeast.as_mut().unwrap().insert(point, index);
                } else {
                    self.northeast.as_mut().unwrap().insert(point, index);
                }
            } else {
                if y >= center_y {
                    self.southwest.as_mut().unwrap().insert(point, index);
                } else {
                    self.northwest.as_mut().unwrap().insert(point, index);
                }
            }
        }
    }

    
    pub fn clear(&mut self) {
        self.indices.clear();
        self.points.clear();
        if self.divided {
            self.northeast.as_mut().unwrap().clear();
            self.northwest.as_mut().unwrap().clear();
            self.southeast.as_mut().unwrap().clear();
            self.southwest.as_mut().unwrap().clear();
        }
        self.divided = false;
    }
    
    
    pub fn subdivide(&mut self){
        self.divided = true;
        let x: f32 = self.boundary.x;
        let y: f32 = self.boundary.y;
        let hh: f32 = self.boundary.h / 2.0;
        let hw: f32 = self.boundary.w / 2.0;

        self.northwest = Some(Box::new(QuadTree::new(Rect::new(x - hw, y - hh, hw, hh), self.capacity)));
        self.northeast = Some(Box::new(QuadTree::new(Rect::new(x + hw, y - hh, hw, hh), self.capacity)));
        self.southwest = Some(Box::new(QuadTree::new(Rect::new(x - hw, y + hh, hw, hh), self.capacity)));
        self.southeast = Some(Box::new(QuadTree::new(Rect::new(x + hw, y + hh, hw, hh), self.capacity)));
    }
    
    pub fn query_cone(
        &self,
        cone_origin: Vector2<f32>,
        direction: Vector2<f32>,
        radius: f32,
        angle: f32,
    ) -> Vec<i32> {
        let mut found: Vec<i32> = Vec::new();
    
        if radius <= 0.0 || angle <= 0.0 {
            return found; // invalid cone
        }
    
        let dir_norm = if direction == Vector2::new(0.0, 0.0) {
            // If direction is zero, use a default direction
            // This could be any arbitrary direction, here we use (1, 0)
            // This avoids division by zero in normalization
            Vector2::new(1.0, 0.0)
        } else {
            direction.normalize()
        };
    
        if !self.boundary.intersects_cone(cone_origin, dir_norm, radius, angle) {
            return found;
        }
    
        for i in 0..self.indices.len() {
            if Rect::point_in_cone(self.points[i], cone_origin, dir_norm, radius, angle) {
                found.push(self.indices[i]);
            }
        }
    
        if self.divided {
            found.extend(self.northwest.as_ref().unwrap().query_cone(cone_origin, dir_norm, radius, angle));
            found.extend(self.northeast.as_ref().unwrap().query_cone(cone_origin, dir_norm, radius, angle));
            found.extend(self.southwest.as_ref().unwrap().query_cone(cone_origin, dir_norm, radius, angle));
            found.extend(self.southeast.as_ref().unwrap().query_cone(cone_origin, dir_norm, radius, angle));
        }
    
        found
    }

    pub fn query(&self, rect: &Rect)->Vec<i32>{
        let mut found: Vec<i32> = Vec::new();
        if !self.boundary.intersects(*rect){
            return found;
        }

        for i in 0..self.indices.len(){
            if rect.point_is_in(self.points[i]){
                found.push(self.indices[i]);
            }
        }

        if(self.divided){
            found.extend(self.northwest.as_ref().unwrap().query(rect));
            found.extend(self.northeast.as_ref().unwrap().query(rect));
            found.extend(self.southwest.as_ref().unwrap().query(rect));
            found.extend(self.southeast.as_ref().unwrap().query(rect));
        }

        return found;
    }
}