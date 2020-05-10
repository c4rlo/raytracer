#[derive(Clone, Copy)]
struct Vec3(f64, f64, f64);

impl Vec3 {
    fn dot(&self, other: &Vec3) -> f64 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

const CANVAS_WIDTH: u32 = 600;
const CANVAS_HEIGHT: u32 = 600;

const CAMERA: Vec3 = Vec3(0., 0., 0.);

const VIEWPORT_WIDTH: f64 = 1.;
const VIEWPORT_HEIGHT: f64 = (CANVAS_HEIGHT as f64) * VIEWPORT_WIDTH / (CANVAS_WIDTH as f64);
const VIEWPORT_DISTANCE: f64 = 1.;

type Colour = image::Rgb<u8>;

const BGCOLOUR: Colour = image::Rgb([255u8, 255u8, 255u8]);

struct Sphere {
    centre: Vec3,
    radius: f64,
    colour: Colour,
}

const SCENE: [Sphere; 3] = [
    Sphere{
        centre: Vec3(0., -1., 3.),
        radius: 1.,
        colour: image::Rgb([255u8, 0u8, 0u8]),
    },
    Sphere{
        centre: Vec3(2., 0., 4.),
        radius: 1.,
        colour: image::Rgb([0u8, 0u8, 255u8]),
    },
    Sphere{
        centre: Vec3(-2., 0., 4.),
        radius: 1.,
        colour: image::Rgb([0u8, 255u8, 0u8]),
    },
];

fn trace_ray(camera: &Vec3, dir: &Vec3, t_min: f64, t_max: f64) -> Colour {
    let mut closest_t = f64::INFINITY;
    let mut closest_sphere = None;
    for sphere in &SCENE {
        let (t1, t2) = intersect_ray_sphere(camera, dir, sphere);
        if (t_min..t_max).contains(&t1) && t1 < closest_t {
            closest_t = t1;
            closest_sphere = Some(sphere);
        }
        if (t_min..t_max).contains(&t2) && t2 < closest_t {
            closest_t = t2;
            closest_sphere = Some(sphere);
        }
    }
    match closest_sphere {
        None => BGCOLOUR,
        Some(sphere) => sphere.colour
    }
}

fn intersect_ray_sphere(camera: &Vec3, dir: &Vec3, sphere: &Sphere) -> (f64, f64) {
    let oc = *camera - sphere.centre;
    let k1 = dir.dot(dir);
    let k2 = 2. * oc.dot(dir);
    let k3 = oc.dot(&oc) - sphere.radius * sphere.radius;

    let discriminant = k2*k2 - 4.*k1*k3;

    if discriminant < 0. {
        (f64::INFINITY, f64::INFINITY)
    } else {
        let sd = discriminant.sqrt();
        let d = 2. * k1;
        ((-k2 + sd) / d, (-k2 - sd) / d)
    }
}

fn canvas_to_viewport(x: u32, y: u32) -> Vec3 {
    let wf = CANVAS_WIDTH as f64;
    let hf = CANVAS_HEIGHT as f64;
    Vec3(
        (x as f64 - wf / 2.) / wf * VIEWPORT_WIDTH,
        (y as f64 - hf / 2.) / hf * VIEWPORT_HEIGHT,
        VIEWPORT_DISTANCE
    )
}

fn main() {
    let mut img = image::RgbImage::new(CANVAS_WIDTH, CANVAS_HEIGHT);
    for x in 0..CANVAS_WIDTH {
        for y in 0..CANVAS_HEIGHT {
            let ray_dir = canvas_to_viewport(x, y);
            let colour = trace_ray(&CAMERA, &ray_dir, 1., f64::INFINITY);
            img.put_pixel(x, CANVAS_HEIGHT - y - 1, colour);
        }
    }
    img.save("img.png").unwrap()
}
