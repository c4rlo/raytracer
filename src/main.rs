// https://www.gabrielgambetta.com/computer-graphics-from-scratch/raytracing.html

#[derive(Clone, Copy, Default)]
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

const BGCOLOUR: Vec3 = Vec3(0., 0., 0.);

struct Sphere {
    centre: Vec3,
    radius: f64,
    colour: Vec3,
    specular: f64,
    reflective: f64,
    transparent: f64,
}

const SPHERES: [Sphere; 5] = [
    Sphere{
        centre: Vec3(0.1, -1., 3.),
        radius: 1.2,
        colour: Vec3(1., 0., 0.),
        specular: 500.,
        reflective: 0.2,
        transparent: 0.5,
    },
    Sphere{
        centre: Vec3(2., 0., 4.),
        radius: 1.,
        colour: Vec3(0., 0., 1.),
        specular: 500.,
        reflective: 0.5,
        transparent: 0.,
    },
    Sphere{
        centre: Vec3(-2., 0., 4.),
        radius: 2.,
        colour: Vec3(0., 1., 0.),
        specular: 10.,
        reflective: 0.4,
        transparent: 0.,
    },
    Sphere{
        centre: Vec3(0., -5001., 0.),
        radius: 5000.,
        colour: Vec3(1., 1., 0.),
        specular: 1000.,
        reflective: 0.1,
        transparent: 0.2,
    },
    Sphere{
        centre: Vec3(2.6, 4.6, 13.),
        radius: 4.,
        colour: Vec3(0.7, 0.7, 1.),
        specular: 0.,
        reflective: 0.7,
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
    fn dot(self, other: Vec3) -> f64 {
        self.0 * other.0 + self.1 * other.1 + self.2 * other.2
    }

    fn pointwise_mul(self, other: Vec3) -> Vec3 {
        Vec3(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn as_unit(self) -> Vec3 {
        self.norm().recip() * self
    }

    fn to_rgb_u8(self) -> image::Rgb<u8> {
        image::Rgb::<u8>([
            channel_to_rgb_u8(self.0),
            channel_to_rgb_u8(self.1),
            channel_to_rgb_u8(self.2),
        ])
    }
}

fn channel_to_rgb_u8(c: f64) -> u8 {
    if c < 0. {
        0u8
    } else if c >= 1. {
        255u8
    } else {
        (c * 256.) as u8
    }
}

impl std::ops::Add for Vec3 {
    type Output = Vec3;

    fn add(self, other: Vec3) -> Vec3 {
        Vec3(self.0 + other.0, self.1 + other.1, self.2 + other.2)
    }
}

impl std::ops::AddAssign for Vec3 {
    fn add_assign(&mut self, rhs: Vec3) {
        self.0 += rhs.0;
        self.1 += rhs.1;
        self.2 += rhs.2;
    }
}

impl std::ops::Sub for Vec3 {
    type Output = Vec3;

    fn sub(self, other: Vec3) -> Vec3 {
        Vec3(self.0 - other.0, self.1 - other.1, self.2 - other.2)
    }
}

impl std::ops::Mul<Vec3> for f64 {
    type Output = Vec3;

    fn mul(self, v: Vec3) -> Vec3 {
        Vec3(self * v.0, self * v.1, self * v.2)
    }
}

impl std::ops::Neg for Vec3 {
    type Output = Vec3;

    fn neg(self) -> Vec3 {
        -1. * self
    }
}


// Returns colour (RGB)
fn trace_ray(origin: Vec3, ray_dir: Vec3, t_min: f64, t_max: f64, recursion: u32) -> Vec3 {
    match closest_intersection(origin, ray_dir, t_min, t_max) {
        None => BGCOLOUR,
        Some((sphere, t)) => {
            let intersection = origin + t * ray_dir;
            let normal = (intersection - sphere.centre).as_unit();
            let light = compute_lighting(intersection, normal, -ray_dir, sphere.specular);
            let local_colour = light.pointwise_mul(sphere.colour);

            let mut reflect_factor = sphere.reflective;
            let mut reflect_colour = Vec3::default();
            if recursion == 0 {
                reflect_factor = 0.;
            }
            if reflect_factor > 0. {
                let reflect = reflect_ray(-ray_dir, normal);
                reflect_colour = trace_ray(intersection, reflect, EPSILON, f64::INFINITY,
                    recursion - 1);
            }

            let mut transparent_colour = Vec3::default();
            if sphere.transparent > 0. {
                transparent_colour = trace_ray(intersection, ray_dir, EPSILON, f64::INFINITY,
                    recursion);
            }

            let local_factor = 1. - reflect_factor - sphere.transparent;
            local_factor * local_colour +
                reflect_factor * reflect_colour +
                sphere.transparent * transparent_colour
        }
    }
}

fn closest_intersection(origin: Vec3, ray_dir: Vec3, t_min: f64, t_max: f64)
        -> Option<(&'static Sphere, f64)> {
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

fn intersect_ray_sphere(origin: Vec3, ray_dir: Vec3, sphere: &Sphere) -> (f64, f64) {
    let oc = origin - sphere.centre;
    let k1 = ray_dir.dot(ray_dir);
    let k2 = 2. * oc.dot(ray_dir);
    let k3 = oc.dot(oc) - sphere.radius * sphere.radius;

    let discriminant = k2*k2 - 4.*k1*k3;

    if discriminant < 0. {
        (f64::INFINITY, f64::INFINITY)
    } else {
        let sd = discriminant.sqrt();
        let d = 2. * k1;
        ((-k2 + sd) / d, (-k2 - sd) / d)
    }
}

// Returns: RGB colour
fn compute_lighting(point: Vec3, normal: Vec3, view: Vec3, specular: f64) -> Vec3 {
    let mut result = Vec3::default();
    for light in &LIGHTS {
        result += light.intensity * match light.light_type {
            LightType::Ambient => Vec3(1., 1., 1.),
            LightType::Point(source) =>
                directional_light(point, normal, view, source - point, 1., specular),
            LightType::Directional(light_dir) =>
                directional_light(point, normal, view, light_dir, f64::INFINITY, specular),
        }
    }
    result
}

// 'point': surface point (measured from origin)
// 'view': vector from surface point to camera
// 'normal': surface normal
// 'light_dir': vector from surface point in direction of light source
// 'light_dist': distance of light from surface point, measured in units of 'light_dir'
// 'specular': specular exponent, or negative if no specular lighting
// Returns: RGB colour
fn directional_light(point: Vec3, normal: Vec3, view: Vec3, light_dir: Vec3, light_dist: f64,
                     specular: f64) -> Vec3 {
    // Shadow check
    if closest_intersection(point, light_dir, EPSILON, light_dist).is_some() {
        // TODO take into account transparency
        return Vec3::default();
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

    intensity * Vec3(1., 1., 1.)
}

fn reflect_ray(ray: Vec3, normal: Vec3) -> Vec3 {
    2. * normal.dot(ray) * normal - ray
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
            let colour = trace_ray(CAMERA, ray_dir, 1., f64::INFINITY, REFLECT_RECURSION);
            img.put_pixel(x, CANVAS_HEIGHT - y - 1, colour.to_rgb_u8());
        }
    }
    img.save("img.png").unwrap()
}
