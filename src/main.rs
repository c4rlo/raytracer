// https://www.gabrielgambetta.com/computer-graphics-from-scratch/raytracing.html

// TODO (approx priority order):
// - Optimization: use all CPU cores
// - Optimization: precompute more things
// - Custom camera position
// - Refractions
// - Depth of field
// - Constructive Solid Geometry (Boolean shapes)
// - Profile to find bottlenecks + optimize
// - Other shapes, e.g. triangles
// - Textures, perhaps programmatically generated, e.g. checkerboard
// - Maybe fractals as textures?!
// - Ability to render standard 3D image formats
// - Stereo vision

#[derive(Clone, Copy, PartialEq, Default, Debug)]
struct Vec3(f64, f64, f64);

impl Vec3 {
    const ZERO: Vec3 = Vec3(0., 0., 0.);
}

const CANVAS_WIDTH: u32 = 810;
const CANVAS_HEIGHT: u32 = 950;

const CANVAS_WIDTH_F64: f64 = CANVAS_WIDTH as f64;
const CANVAS_HEIGHT_F64: f64 = CANVAS_HEIGHT as f64;

const CAMERA: Vec3 = Vec3(0., 0., 0.);

const VIEWPORT_WIDTH: f64 = 1.;
const VIEWPORT_HEIGHT: f64 = CANVAS_HEIGHT_F64 * VIEWPORT_WIDTH / CANVAS_WIDTH_F64;
const VIEWPORT_DISTANCE: f64 = 1.;

const REFLECT_RECURSION: u32 = 5;

const EPSILON: f64 = 0.000000001;

#[derive(Debug)]
struct Sphere {
    centre: Vec3,
    radius_sq: f64,
    colour: Vec3,
    specular: f64,
    reflective: f64,
    transparent: f64,
}

#[derive(Debug)]
enum LightType {
    Ambient,
    Directional(Vec3),
    Point(Vec3),
}

#[derive(Debug)]
struct Light {
    intensity: f64,
    light_type: LightType,
}

#[derive(Debug)]
struct Scene {
    bgcolour: Vec3,
    spheres: Vec<Sphere>,
    lights: Vec<Light>,
}

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
        (c * 255.) as u8
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
fn trace_ray(
    scene: &Scene,
    origin: Vec3,
    ray_dir: Vec3,
    t_min: f64,
    t_max: f64,
    recursion: u32,
) -> Vec3 {
    match closest_intersection(&scene.spheres, origin, ray_dir, t_min, t_max) {
        None => scene.bgcolour,
        Some((sphere, t)) => {
            let intersection = origin + t * ray_dir;
            let normal = (intersection - sphere.centre).as_unit();
            let light = compute_lighting(&scene, intersection, normal, -ray_dir, sphere.specular);
            let local_colour = light.pointwise_mul(sphere.colour);

            let mut reflect_factor = sphere.reflective;
            let mut reflect_colour = Vec3::default();
            if recursion == 0 {
                reflect_factor = 0.;
            }
            if reflect_factor > 0. {
                let reflect = reflect_ray(-ray_dir, normal);
                reflect_colour = trace_ray(
                    scene,
                    intersection,
                    reflect,
                    EPSILON,
                    f64::INFINITY,
                    recursion - 1,
                );
            }

            let mut transparent_colour = Vec3::default();
            if sphere.transparent > 0. {
                transparent_colour = trace_ray(
                    scene,
                    intersection,
                    ray_dir,
                    EPSILON,
                    f64::INFINITY,
                    recursion,
                );
            }

            let local_factor = 1. - reflect_factor - sphere.transparent;
            local_factor * local_colour
                + reflect_factor * reflect_colour
                + sphere.transparent * transparent_colour
        }
    }
}

fn closest_intersection(
    spheres: &[Sphere],
    origin: Vec3,
    ray_dir: Vec3,
    t_min: f64,
    t_max: f64,
) -> Option<(&Sphere, f64)> {
    let mut closest_t = f64::INFINITY;
    let mut closest_sphere = None;
    for (sphere, t) in intersected_spheres(spheres, origin, ray_dir, t_min, t_max) {
        if t < closest_t {
            closest_t = t;
            closest_sphere = Some(sphere);
        }
    }
    match closest_sphere {
        None => None,
        Some(sphere) => Some((sphere, closest_t)),
    }
}

struct SpheresIterator<'a> {
    // Inputs
    spheres: &'a [Sphere],
    ray_origin: Vec3,
    ray_dir: Vec3,
    t_range: std::ops::Range<f64>,

    // State
    k1: f64,
    sphere_index: isize,
    t: Option<f64>,
}

impl<'a> SpheresIterator<'a> {
    fn curr_item(&self, t: f64) -> Option<(&'a Sphere, f64)> {
        Some((self.curr_sphere(), t))
    }

    fn curr_sphere(&self) -> &'a Sphere {
        &self.spheres[self.sphere_index as usize]
    }
}

impl<'a> Iterator for SpheresIterator<'a> {
    type Item = (&'a Sphere, f64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.t.is_none() {
            self.sphere_index += 1;
            if self.sphere_index as usize >= self.spheres.len() {
                return None;
            }
            let (t1, t2) =
                intersect_ray_sphere(self.ray_origin, self.ray_dir, self.curr_sphere(), self.k1);
            if self.t_range.contains(&t1) {
                self.t = Some(t1);
            }
            if self.t_range.contains(&t2) {
                return self.curr_item(t2);
            }
        }
        let t = self.t.take().unwrap();
        self.curr_item(t)
    }
}

fn intersected_spheres<'a>(
    spheres: &'a [Sphere],
    origin: Vec3,
    ray_dir: Vec3,
    t_min: f64,
    t_max: f64,
) -> impl Iterator<Item = (&'a Sphere, f64)> {
    SpheresIterator {
        spheres: spheres,
        ray_origin: origin,
        ray_dir: ray_dir,
        t_range: t_min..t_max,
        k1: ray_dir.dot(ray_dir),
        sphere_index: -1,
        t: None,
    }
}

fn intersect_ray_sphere(origin: Vec3, ray_dir: Vec3, sphere: &Sphere, k1: f64) -> (f64, f64) {
    let oc = origin - sphere.centre;
    // let k1 = ray_dir.dot(ray_dir);  <- we get this from our caller - optimization
    let k2 = 2. * oc.dot(ray_dir);
    // let k3 = oc.dot(oc) - sphere.radius * sphere.radius;
    let k3 = oc.dot(oc) - sphere.radius_sq;

    let discriminant = k2 * k2 - 4. * k1 * k3;

    if discriminant < 0. {
        (f64::INFINITY, f64::INFINITY)
    } else {
        let sd = discriminant.sqrt();
        let d = 2. * k1;
        ((-k2 + sd) / d, (-k2 - sd) / d)
    }
}

// Returns: RGB colour
fn compute_lighting(scene: &Scene, point: Vec3, normal: Vec3, view: Vec3, specular: f64) -> Vec3 {
    let mut result = Vec3::ZERO;
    for light in &scene.lights {
        let colour = match light.light_type {
            LightType::Ambient => Vec3(1., 1., 1.),
            LightType::Point(source) => directional_light(
                point,
                normal,
                view,
                source - point,
                1.,
                specular,
                &scene.spheres,
            ),
            LightType::Directional(light_dir) => directional_light(
                point,
                normal,
                view,
                light_dir,
                f64::INFINITY,
                specular,
                &scene.spheres,
            ),
        };
        result += light.intensity * colour;
    }
    result
}

// Returns: RGB colour
fn directional_light(
    point: Vec3,        // surface point (measured from origin)
    normal: Vec3,       // surface normal
    view: Vec3,         // vector from surface point to camera
    light_dir: Vec3,    // vector from surface point in direction of light source
    light_dist: f64,    // distance of light from surface point, measured in units of 'light_dir'
    specular: f64,      // specular exponent, or negative if no specular lighting
    spheres: &[Sphere], // spheres in the scene (for shadow calculation)
) -> Vec3 {
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

    if intensity == 0. {
        return Vec3::ZERO;
    }

    intensity * shadow(spheres, point, light_dir, EPSILON, light_dist)
}

// Returns: RGB colour
fn shadow(spheres: &[Sphere], origin: Vec3, ray_dir: Vec3, t_min: f64, t_max: f64) -> Vec3 {
    let mut colour = Vec3(1., 1., 1.);
    for (sphere, _) in intersected_spheres(spheres, origin, ray_dir, t_min, t_max) {
        colour = colour.pointwise_mul(sphere.transparent * sphere.colour);

        // Or if we don't care about the colour, we could do:
        // colour = sphere.transparent * colour

        // Or if we want shadows un-influenced by transparency, we could do:
        // return Vec3::ZERO;

        if colour == Vec3::ZERO {
            break;
        }
    }

    // Note: we taken into account the _transparency_ of the spheres along the ray; really what's
    // missing is their _reflectivity_. However, this seems much harder to do (the route between
    // the origin and the light source is no longer a simple ray).

    colour
}

fn reflect_ray(ray: Vec3, normal: Vec3) -> Vec3 {
    2. * normal.dot(ray) * normal - ray
}

fn canvas_to_viewport(x: u32, y: u32) -> Vec3 {
    Vec3(
        (x as f64 - CANVAS_WIDTH_F64 / 2.) / CANVAS_WIDTH_F64 * VIEWPORT_WIDTH,
        (y as f64 - CANVAS_HEIGHT_F64 / 2.) / CANVAS_HEIGHT_F64 * VIEWPORT_HEIGHT,
        VIEWPORT_DISTANCE,
    )
}

fn parse_scene(description: &[u8]) -> Scene {
    let v: serde_hjson::Value = serde_hjson::from_slice(description).unwrap();
    Scene {
        bgcolour: parse_colour(v.find("bgcolour").unwrap()),
        spheres: v
            .find("spheres")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(parse_sphere)
            .collect(),
        lights: v
            .find("lights")
            .unwrap()
            .as_array()
            .unwrap()
            .iter()
            .map(parse_light)
            .collect(),
    }
}

fn parse_sphere(v: &serde_hjson::Value) -> Sphere {
    Sphere {
        centre: parse_vec3(v.find("centre").unwrap()),
        radius_sq: v.find("radius").unwrap().as_f64().unwrap().powi(2),
        colour: parse_colour(v.find("colour").unwrap()),
        specular: parse_opt_f64(v.find("specular")),
        reflective: parse_opt_f64(v.find("reflective")),
        transparent: parse_opt_f64(v.find("transparent")),
    }
}

fn parse_light(v: &serde_hjson::Value) -> Light {
    Light {
        intensity: v.find("intensity").unwrap().as_f64().unwrap(),
        light_type: parse_light_type(v),
    }
}

fn parse_light_type(v: &serde_hjson::Value) -> LightType {
    match v.find("type").unwrap().as_str().unwrap() {
        "ambient" => LightType::Ambient,
        "directional" => LightType::Directional(parse_vec3(v.find("dir").unwrap())),
        "point" => LightType::Point(parse_vec3(v.find("pos").unwrap())),
        _ => panic!("bad light type"),
    }
}

fn parse_vec3(v: &serde_hjson::Value) -> Vec3 {
    let a = v.as_array().unwrap();
    Vec3(
        a[0].as_f64().unwrap(),
        a[1].as_f64().unwrap(),
        a[2].as_f64().unwrap(),
    )
}

fn parse_opt_f64(vo: Option<&serde_hjson::Value>) -> f64 {
    vo.map_or(0., |v| v.as_f64().unwrap())
}

fn parse_colour(v: &serde_hjson::Value) -> Vec3 {
    let s = v.as_str().unwrap();
    Vec3(
        parse_colour_channel(&s[0..2]),
        parse_colour_channel(&s[2..4]),
        parse_colour_channel(&s[4..6]),
    )
}

fn parse_colour_channel(s: &str) -> f64 {
    u8::from_str_radix(&s, 16).unwrap() as f64 / 255.
}

fn main() {
    let mut args = std::env::args().skip(1);
    let in_file = args.next().unwrap();
    let out_file = args.next().unwrap();
    let scene = parse_scene(&std::fs::read(in_file).unwrap());

    let mut img = image::RgbImage::new(CANVAS_WIDTH, CANVAS_HEIGHT);
    for x in 0..CANVAS_WIDTH {
        for y in 0..CANVAS_HEIGHT {
            let ray_dir = canvas_to_viewport(x, y);
            let colour = trace_ray(
                &scene,
                CAMERA,
                ray_dir,
                1.,
                f64::INFINITY,
                REFLECT_RECURSION,
            );
            img.put_pixel(x, CANVAS_HEIGHT - y - 1, colour.to_rgb_u8());
        }
    }
    img.save(out_file).unwrap()
}
