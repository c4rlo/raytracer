#[derive(Clone, Copy)]
struct Vec3(f64, f64, f64);

const CANVAS_WIDTH: u32 = 800;
const CANVAS_HEIGHT: u32 = 950;

const CANVAS_WIDTH_F64: f64 = CANVAS_WIDTH as f64;
const CANVAS_HEIGHT_F64: f64 = CANVAS_HEIGHT as f64;

const CAMERA: Vec3 = Vec3(0., 0., 0.);

const VIEWPORT_WIDTH: f64 = 1.;
const VIEWPORT_HEIGHT: f64 = CANVAS_HEIGHT_F64 * VIEWPORT_WIDTH / CANVAS_WIDTH_F64;
const VIEWPORT_DISTANCE: f64 = 1.;

const REFLECT_RECURSION: u32 = 5;

const EPSILON: f64 = 0.000000001;

const BGCOLOUR: image::Rgb<u8> = image::Rgb([0u8, 0u8, 0u8]);

struct Sphere {
    centre: Vec3,
    radius: f64,
    colour: image::Rgb<u8>,
    specular: f64,
    reflective: f64,
    transparent: f64,
}

const SPHERES: [Sphere; 5] = [
    Sphere{
        centre: Vec3(0.1, -1., 3.),
        radius: 1.2,
        colour: image::Rgb([255u8, 0u8, 0u8]),
        specular: 500.,
        reflective: 0.2,
        transparent: 0.5,
    },
    Sphere{
        centre: Vec3(2., 0., 4.),
        radius: 1.,
        colour: image::Rgb([0u8, 0u8, 255u8]),
        specular: 500.,
        reflective: 0.5,
        transparent: 0.,
    },
    Sphere{
        centre: Vec3(-2., 0., 4.),
        radius: 2.,
        colour: image::Rgb([0u8, 255u8, 0u8]),
        specular: 10.,
        reflective: 0.4,
        transparent: 0.,
    },
    Sphere{
        centre: Vec3(0., -5001., 0.),
        radius: 5000.,
        colour: image::Rgb([255u8, 255u8, 0u8]),
        specular: 1000.,
        reflective: 0.1,
        transparent: 0.2,
    },
    Sphere{
        centre: Vec3(2.6, 4.6, 13.),
        radius: 4.,
        colour: image::Rgb([200u8, 200u8, 255u8]),
        specular: 0.1,
        reflective: 0.9,
        transparent: 0.0,
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


fn trace_ray(origin: &Vec3, ray_dir: &Vec3, t_min: f64, t_max: f64, recursion: u32) -> image::Rgb<u8> {
    match closest_intersection(origin, ray_dir, t_min, t_max) {
        None => BGCOLOUR,
        Some((sphere, t)) => {
            let intersection = origin + &(t * ray_dir);
            let normal = (&intersection - &sphere.centre).as_unit();
            let intensity = compute_lighting(&intersection, &normal, &-ray_dir,
                sphere.specular);
            let local_colour = colour_mul(intensity, sphere.colour);

            let mut reflect_factor = sphere.reflective;
            let mut reflect_colour = image::Rgb::<u8>([0u8, 0u8, 0u8]);
            if recursion == 0 {
                reflect_factor = 0.;
            }
            if reflect_factor > 0. {
                let reflect = reflect_ray(&-ray_dir, &normal);
                reflect_colour = trace_ray(&intersection, &reflect, EPSILON, f64::INFINITY,
                    recursion - 1);
            }

            let mut transparent_colour = image::Rgb::<u8>([0u8, 0u8, 0u8]);
            if sphere.transparent > 0. {
                transparent_colour = trace_ray(&intersection, ray_dir, EPSILON, f64::INFINITY, recursion);
            }

            colour_mix(local_colour, reflect_colour, reflect_factor, transparent_colour, sphere.transparent)
        }
    }
}

fn closest_intersection(origin: &Vec3, ray_dir: &Vec3, t_min: f64, t_max: f64) -> Option<(&'static Sphere, f64)> {
    let mut closest_t = f64::INFINITY;
    let mut closest_sphere = None;
    for sphere in &SPHERES {
        let (t1, t2) = intersect_ray_sphere(origin, ray_dir, sphere);
        if t1 < closest_t && (t_min..t_max).contains(&t1) {
            closest_t = t1;
            closest_sphere = Some(sphere);
        }
        if t2 < closest_t && (t_min..t_max).contains(&t2) {
            closest_t = t2;
            closest_sphere = Some(sphere);
        }
    }
    match closest_sphere {
        None => None,
        Some(sphere) => Some((sphere, closest_t)),
    }
}

fn intersect_ray_sphere(origin: &Vec3, ray_dir: &Vec3, sphere: &Sphere) -> (f64, f64) {
    let oc = origin - &sphere.centre;
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
                directional_light(point, normal, view, &(&source - point), 1., specular),
            LightType::Directional(light_dir) =>
                directional_light(point, normal, view, &light_dir, f64::INFINITY, specular),
        }
    }
    intensity
}

// 'point': surface point (measured from origin)
// 'view': vector from surface point to camera
// 'normal': surface normal
// 'light_dir': vector from surface point in direction of light source
// 'light_dist': distance of light from surface point, measured in units of 'light_dir'
// 'specular': specular exponent, or negative if no specular lighting
fn directional_light(point: &Vec3, normal: &Vec3, view: &Vec3, light_dir: &Vec3, light_dist: f64,
                     specular: f64) -> f64 {
    // Shadow check
    if closest_intersection(point, light_dir, EPSILON, light_dist).is_some() {
        return 0.;
    }

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
        let reflect = reflect_ray(light_dir, normal);
        let b = reflect.dot(view);
        if b > 0. {
            intensity += ((reflect.norm() * view.norm()).recip() * b).powf(specular);
        }
    }

    intensity
}

fn colour_mul(f: f64, c: image::Rgb<u8>) -> image::Rgb<u8> {
    image::Rgb::<u8>([
        f64_to_u8(f * (c[0] as f64)),
        f64_to_u8(f * (c[1] as f64)),
        f64_to_u8(f * (c[2] as f64)),
    ])
}

fn colour_mix(c0: image::Rgb<u8>, c1: image::Rgb<u8>, f1: f64, c2: image::Rgb<u8>, f2: f64) -> image::Rgb<u8> {
    let f0 = 1. - f1 - f2;
    image::Rgb::<u8>([
        f64_to_u8(f0 * (c0[0] as f64) + f1 * (c1[0] as f64) + f2 * (c2[0] as f64)),
        f64_to_u8(f0 * (c0[1] as f64) + f1 * (c1[1] as f64) + f2 * (c2[1] as f64)),
        f64_to_u8(f0 * (c0[2] as f64) + f1 * (c1[2] as f64) + f2 * (c2[2] as f64)),
    ])
}

fn f64_to_u8(f: f64) -> u8 {
    if f < 0. {
        0u8
    } else if f >= 256. {
        255u8
    } else {
        f as u8
    }
}

fn reflect_ray(ray: &Vec3, normal: &Vec3) -> Vec3 {
    &(2. * normal.dot(ray) * normal) - ray
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
            let colour = trace_ray(&CAMERA, &ray_dir, 1., f64::INFINITY, REFLECT_RECURSION);
            img.put_pixel(x, CANVAS_HEIGHT - y - 1, colour);
        }
    }
    img.save("img.png").unwrap()
}
