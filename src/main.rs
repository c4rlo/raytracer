#[derive(Clone, Copy)]
struct Vec3(f64, f64, f64);

impl Vec3 {
    fn dot(&self, other: &Vec3) -> f64 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }

    fn norm(&self) -> f64 {
        self.dot(self).sqrt()
    }

    fn as_unit(&self) -> Vec3 {
        self.norm().recip() * self
    }
}

impl std::ops::Add for &Vec3 {
    type Output = Vec3;

    fn add(self, other: &Vec3) -> Vec3 {
        Vec3(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl std::ops::Sub for &Vec3 {
    type Output = Vec3;

    fn sub(self, other: &Vec3) -> Vec3 {
        Vec3(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl std::ops::Mul<&Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, v: &Vec3) -> Vec3 {
        Vec3(self * v.0, self * v.1, self * v.2)
    }
}

impl std::ops::Neg for &Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        -1. * self
    }
}

const CANVAS_WIDTH: u32 = 800;
const CANVAS_HEIGHT: u32 = 950;

const CANVAS_WIDTH_F64: f64 = CANVAS_WIDTH as f64;
const CANVAS_HEIGHT_F64: f64 = CANVAS_HEIGHT as f64;

const CAMERA: Vec3 = Vec3(0., 0., 0.);

const VIEWPORT_WIDTH: f64 = 1.;
const VIEWPORT_HEIGHT: f64 = CANVAS_HEIGHT_F64 * VIEWPORT_WIDTH / CANVAS_WIDTH_F64;
const VIEWPORT_DISTANCE: f64 = 1.;

type Colour = image::Rgb<u8>;

fn colour_channel_mul(f: f64, c: u8) -> u8 {
    let raw = f * (c as f64);
    if raw < 0. {
        0u8
    } else if raw >= 256. {
        255u8
    } else {
        raw as u8
    }
}

fn colour_mul(f: f64, c: image::Rgb<u8>) -> image::Rgb<u8> {
    image::Rgb::<u8>([
        colour_channel_mul(f, c[0]),
        colour_channel_mul(f, c[1]),
        colour_channel_mul(f, c[2]),
    ])
}

const BGCOLOUR: Colour = image::Rgb([255u8, 255u8, 255u8]);

struct Sphere {
    centre: Vec3,
    radius: f64,
    colour: Colour,
    specular: f64,
}

const SPHERES: [Sphere; 4] = [
    Sphere{
        centre: Vec3(0., -1., 3.),
        radius: 1.,
        colour: image::Rgb([255u8, 0u8, 0u8]),
        specular: 500.,
    },
    Sphere{
        centre: Vec3(2., 0., 4.),
        radius: 1.,
        colour: image::Rgb([0u8, 0u8, 255u8]),
        specular: 500.,
    },
    Sphere{
        centre: Vec3(-2., 0., 4.),
        radius: 1.,
        colour: image::Rgb([0u8, 255u8, 0u8]),
        specular: 10.,
    },
    Sphere{
        centre: Vec3(0., -5001., 0.),
        radius: 5000.,
        colour: image::Rgb([255u8, 255u8, 0u8]),
        specular: 1000.,
    },
];

enum LightType {
    Ambient,
    Directional(Vec3),
    Point(Vec3),
}

struct Light {
    intensity: f64,
    light_type: LightType,
}

const LIGHTS: [Light; 3] = [
    Light{
        intensity: 0.2,
        light_type: LightType::Ambient,
    },
    Light{
        intensity: 0.6,
        light_type: LightType::Point(Vec3(2., 1., 0.)),
    },
    Light{
        intensity: 0.2,
        light_type: LightType::Directional(Vec3(1., 4., 4.)),
    },
];

fn trace_ray(camera: &Vec3, ray_dir: &Vec3, t_min: f64, t_max: f64) -> Colour {
    let mut closest_t = f64::INFINITY;
    let mut closest_sphere = None;
    for sphere in &SPHERES {
        let (t1, t2) = intersect_ray_sphere(camera, ray_dir, sphere);
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
        Some(sphere) => {
            let intersection = camera + &(closest_t * ray_dir);
            let normal = &intersection - &sphere.centre;
            let intensity = compute_lighting(&intersection, &normal.as_unit(), &-ray_dir,
                sphere.specular);
            colour_mul(intensity, sphere.colour)
        }
    }
}

fn intersect_ray_sphere(camera: &Vec3, ray_dir: &Vec3, sphere: &Sphere) -> (f64, f64) {
    let oc = camera - &sphere.centre;
    let k1 = ray_dir.dot(ray_dir);
    let k2 = 2. * oc.dot(ray_dir);
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

fn compute_lighting(point: &Vec3, normal: &Vec3, view: &Vec3, specular: f64) -> f64 {
    let mut intensity = 0.;
    for light in &LIGHTS {
        intensity += light.intensity * match light.light_type {
            LightType::Ambient => 1.,
            LightType::Point(source) =>
                directional_light(view, normal, &(&source - point), specular),
            LightType::Directional(light_dir) =>
                directional_light(view, normal, &light_dir, specular),
        }
    }
    intensity
}

fn directional_light(view: &Vec3, normal: &Vec3, light_dir: &Vec3, specular: f64) -> f64 {
    let mut intensity = 0.;

    // Diffuse lighting
    let a = normal.dot(light_dir);
    if a > 0. {
        // We want:
        //   a / (normal.norm() * light_dir.norm())
        // But 'normal' is a unit normal, so we can simplify to:
        intensity += a / light_dir.norm();
    }

    // Specular lighting
    if specular > 0. {
        let reflect = &(2. * normal.dot(light_dir) * normal) - light_dir;
        let b = reflect.dot(view);
        if b > 0. {
            intensity += ((reflect.norm() * view.norm()).recip() * b).powf(specular);
        }
    }

    intensity
}

fn canvas_to_viewport(x: u32, y: u32) -> Vec3 {
    Vec3(
        (x as f64 - CANVAS_WIDTH_F64 / 2.) / CANVAS_WIDTH_F64 * VIEWPORT_WIDTH,
        (y as f64 - CANVAS_HEIGHT_F64 / 2.) / CANVAS_HEIGHT_F64 * VIEWPORT_HEIGHT,
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
