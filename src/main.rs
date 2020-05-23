use std::ops::Range;

use rayon::prelude::*;

#[derive(Clone, Copy, PartialEq, Default, Debug)]
struct Vec3(f64, f64, f64);

impl Vec3 {
    const ZERO: Vec3 = Vec3(0., 0., 0.);
    const ONES: Vec3 = Vec3(1., 1., 1.);
}

const CANVAS_WIDTH: u32 = 810;
const CANVAS_HEIGHT: u32 = 950;

const CANVAS_WIDTH_F64: f64 = CANVAS_WIDTH as f64;
const CANVAS_HEIGHT_F64: f64 = CANVAS_HEIGHT as f64;

const CAMERA: Vec3 = Vec3(0., 0., 0.);

const VIEWPORT_WIDTH: f64 = 1.;
const VIEWPORT_HEIGHT: f64 = CANVAS_HEIGHT_F64 * VIEWPORT_WIDTH / CANVAS_WIDTH_F64;
const VIEWPORT_DISTANCE: f64 = 1.;

const MAX_RECURSION: u32 = 5;

const EPSILON: f64 = 0.00000000001;

#[derive(Debug)]
struct Sphere {
    centre: Vec3,
    radius_sq: f64,
    colour_ext: Vec3,
    colour_int: Vec3,
    specular: f64,
    reflective: f64,
    refractive: f64,
    transparent: f64,
    priority: u32,
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

    fn norm(self) -> f64 {
        self.dot(self).sqrt()
    }

    fn as_unit(self) -> Vec3 {
        self.norm().recip() * self
    }

    fn powf(self, e: f64) -> Vec3 {
        Vec3(self.0.powf(e), self.1.powf(e), self.2.powf(e))
    }

    fn pointwise_mul(self, other: Vec3) -> Vec3 {
        Vec3(self.0 * other.0, self.1 * other.1, self.2 * other.2)
    }

    fn to_rgb_u8(self) -> image::Rgb<u8> {
        image::Rgb::<u8>([
            channel_to_rgb_u8(self.0),
            channel_to_rgb_u8(self.1),
            channel_to_rgb_u8(self.2),
        ])
    }

    fn assert_ok(self) {
        debug_assert!(!self.0.is_nan());
        debug_assert!(!self.1.is_nan());
        debug_assert!(!self.2.is_nan());
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

impl Sphere {
    fn intersect_ray(&self, origin: Vec3, ray_dir: Vec3, k1: f64) -> (f64, f64) {
        let oc = origin - self.centre;
        let k2 = 2. * oc.dot(ray_dir);
        let k3 = oc.dot(oc) - self.radius_sq;

        let discriminant = k2 * k2 - 4. * k1 * k3;

        if discriminant < 0. {
            (f64::INFINITY, f64::INFINITY)
        } else {
            let sd = discriminant.sqrt();
            let d = 2. * k1;
            ((-k2 - sd) / d, (-k2 + sd) / d)
        }
    }

    // fn contains(&self, point: &Vec3) -> bool {
    //     let a = *point - self.centre;
    //     a.dot(a) <= self.radius_sq
    // }
}

enum RayAction {
    Enter,
    Exit,
    Touch,
}

struct SpheresIterator<'a> {
    // Inputs
    spheres: &'a [Sphere],
    ray_origin: Vec3,
    ray_dir: Vec3,
    t_range: Range<f64>,

    // State
    k1: f64,
    sphere_index: isize,
    backlog: Option<(RayAction, f64)>,
}

impl<'a> SpheresIterator<'a> {
    fn curr_sphere(&self) -> &'a Sphere {
        &self.spheres[self.sphere_index as usize]
    }
}

impl<'a> Iterator for SpheresIterator<'a> {
    type Item = (&'a Sphere, RayAction, f64);

    fn next(&mut self) -> Option<Self::Item> {
        while self.backlog.is_none() {
            self.sphere_index += 1;
            if self.sphere_index as usize >= self.spheres.len() {
                return None;
            }
            let (t1, t2) = self
                .curr_sphere()
                .intersect_ray(self.ray_origin, self.ray_dir, self.k1);
            debug_assert!(!t1.is_nan());
            debug_assert!(!t2.is_nan());
            let (a1, a2) = match t1.partial_cmp(&t2).unwrap() {
                std::cmp::Ordering::Equal => (RayAction::Touch, RayAction::Touch),
                std::cmp::Ordering::Less => (RayAction::Enter, RayAction::Exit),
                std::cmp::Ordering::Greater => (RayAction::Exit, RayAction::Enter),
            };
            if self.t_range.contains(&t1) {
                self.backlog = Some((a1, t1));
            }
            if self.t_range.contains(&t2) {
                return Some((self.curr_sphere(), a2, t2));
            }
        }
        let (a, t) = self.backlog.take().unwrap();
        Some((self.curr_sphere(), a, t))
    }
}

fn intersected_spheres<'a>(
    spheres: &'a [Sphere],
    origin: Vec3,
    ray_dir: Vec3,
    t_range: Range<f64>,
) -> impl Iterator<Item = (&'a Sphere, RayAction, f64)> {
    origin.assert_ok();
    ray_dir.assert_ok();
    SpheresIterator {
        spheres: spheres,
        ray_origin: origin,
        ray_dir: ray_dir,
        t_range: t_range,
        k1: ray_dir.dot(ray_dir),
        sphere_index: -1,
        backlog: None,
    }
}

fn closest_intersection(
    spheres: &[Sphere],
    origin: Vec3,
    ray_dir: Vec3,
    t_range: Range<f64>,
) -> Option<(&Sphere, RayAction, f64)> {
    intersected_spheres(spheres, origin, ray_dir, t_range)
        .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap())
}

#[derive(Clone, Debug)]
struct SphereSet<'a> {
    spheres: smallvec::SmallVec<[&'a Sphere; 32]>,
}

impl<'a> SphereSet<'a> {
    fn new() -> Self {
        Self {
            spheres: smallvec::SmallVec::new(),
        }
    }

    fn add(&mut self, sphere: &'a Sphere) {
        // debug_assert!(self.spheres.iter().find(|&&s| std::ptr::eq(s, sphere)).is_none(),
        //     "spheres={:?} sphere={:?}", self.spheres, sphere);
        #[cfg(debug_assertions)]
        if self
            .spheres
            .iter()
            .find(|&&s| std::ptr::eq(s, sphere))
            .is_some()
        {
            stats::record_double_add();
        }
        self.spheres.push(sphere);
    }

    fn remove(&mut self, sphere: &'a Sphere) {
        for (i, &s) in self.spheres.iter().enumerate() {
            if std::ptr::eq(s, sphere) {
                debug_assert!(std::ptr::eq(self.spheres[i], sphere));
                self.spheres.remove(i);
                return;
            }
        }
        stats::record_double_remove();
        // panic!("{:?} not in list of {} spheres", sphere, self.spheres.len());
    }

    fn adjust(&mut self, action: RayAction, sphere: &'a Sphere) {
        match action {
            RayAction::Enter => {
                self.add(sphere);
            }
            RayAction::Exit => {
                self.remove(sphere);
            }
            _ => {}
        }
    }

    fn adjusted(&self, action: RayAction, sphere: &'a Sphere) -> Self {
        let mut copy = self.clone();
        copy.adjust(action, sphere);
        copy
    }

    fn top_priority_sphere(&self) -> Option<&'a Sphere> {
        self.spheres
            .iter()
            .min_by_key(|sphere| sphere.priority)
            .map(|&s| s)
    }
}

// Returns colour (RGB)
fn trace_ray<'a>(
    scene: &'a Scene,
    origin: Vec3,
    ray_dir: Vec3,
    t_range: Range<f64>,
    in_spheres: &SphereSet<'a>,
    recursion: u32,
) -> Vec3 {
    stats::record_ray();
    match closest_intersection(&scene.spheres, origin, ray_dir, t_range) {
        None => scene.bgcolour,
        Some((sphere, action, t)) => {
            let intersection = origin + t * ray_dir;
            let mut normal = (intersection - sphere.centre).as_unit();
            if let RayAction::Exit = action {
                normal = -normal;
            }

            let local_light =
                compute_lighting(&scene, intersection, normal, -ray_dir, sphere, in_spheres);
            let mut colour = (1. - sphere.transparent - sphere.reflective)
                * local_light.pointwise_mul(sphere.colour_ext);

            let old_top_sphere = in_spheres.top_priority_sphere();

            if recursion > 0 {
                if sphere.transparent > 0. {
                    let in_spheres_new = in_spheres.adjusted(action, sphere);
                    if let Some(refracted_ray) = refract_ray(
                        ray_dir,
                        normal,
                        old_top_sphere,
                        in_spheres_new.top_priority_sphere(),
                    ) {
                        stats::record_refract_ray();
                        colour += sphere.transparent
                            * trace_ray(
                                scene,
                                intersection,
                                refracted_ray,
                                EPSILON..f64::INFINITY,
                                &in_spheres_new,
                                recursion - 1,
                            );
                    }
                }

                if sphere.reflective > 0. {
                    stats::record_reflect_ray();
                    colour += sphere.reflective
                        * trace_ray(
                            scene,
                            intersection,
                            reflect_ray(-ray_dir, normal),
                            EPSILON..f64::INFINITY,
                            in_spheres,
                            recursion - 1,
                        );
                }
            }

            pass_colour_thru_sphere(colour, old_top_sphere, t * ray_dir.norm())
        }
    }
}

// Returns: RGB colour
fn compute_lighting<'a>(
    scene: &'a Scene,
    point: Vec3,
    normal: Vec3,
    view: Vec3,
    sphere: &'a Sphere,
    in_spheres: &SphereSet<'a>,
) -> Vec3 {
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
                sphere,
                &scene.spheres,
                in_spheres,
            ),
            LightType::Directional(light_dir) => directional_light(
                point,
                normal,
                view,
                light_dir,
                f64::INFINITY,
                sphere,
                &scene.spheres,
                in_spheres,
            ),
        };
        result += light.intensity * colour;
    }
    result
}

// Returns: RGB colour
fn directional_light<'a>(
    point: Vec3,        // surface point (measured from origin)
    normal: Vec3,       // surface normal
    view: Vec3,         // vector from surface point to camera/origin
    light_dir: Vec3,    // vector from surface point in direction of light source
    light_dist: f64,    // distance of light from surface point, measured in units of 'light_dir'
    sphere: &'a Sphere,      // sphere on whose surface the point lies
    spheres: &'a [Sphere], // all spheres in the scene (for shadow calculation)
    in_spheres: &SphereSet<'a>,  // spheres containing the surface point (excl sphere itself)
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
    if sphere.specular > 0. {
        let reflect = reflect_ray(light_dir, normal);
        let b = reflect.dot(view);
        if b > 0. {
            intensity += ((reflect.norm() * view.norm()).recip() * b).powf(sphere.specular);
        }
    }

    if intensity == 0. {
        return Vec3::ZERO;
    }

    let mut in_spheres_new = in_spheres.clone();
    if light_dir.dot(normal) < 0. {
        in_spheres_new.add(sphere);
    }

    intensity * shadow(spheres, point, light_dir, EPSILON..light_dist, in_spheres_new)
}

// Returns: RGB colour
// Note: we taken into account the _transparency_ of the spheres along the ray; really what's
// missing is their _reflectivity_. However, this seems much harder to do (the route between
// the origin and the light source is no longer a simple ray).
fn shadow<'a>(
    spheres: &'a [Sphere],
    origin: Vec3,
    ray_dir: Vec3,
    t_range: Range<f64>,
    mut in_spheres: SphereSet<'a>,
) -> Vec3 {

    // All-or-nothing shadows: crap alternative solution.
    // if closest_intersection(spheres, origin, ray_dir, t_range).is_some() {
    //     Vec3::ZERO
    // } else {
    //     Vec3::ONES
    // }

    let ray_norm = ray_dir.norm();

    let adjust_colour = |c: Vec3, sphere: Option<&Sphere>, delta_t: f64| -> Vec3 {
        pass_colour_thru_sphere(c, sphere, delta_t * ray_norm)
    };

    let mut intersections: Vec<_> = intersected_spheres(spheres, origin, ray_dir, t_range.clone())
        .collect();
    intersections.sort_unstable_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    let mut colour = Vec3::ONES;
    let mut t_prev = t_range.start;
    for (sphere, action, t) in intersections {
        colour = adjust_colour(colour, in_spheres.top_priority_sphere(), t - t_prev);
        in_spheres.adjust(action, sphere);
        t_prev = t;

        if colour == Vec3::ZERO {
            return colour;
        }
    }

    adjust_colour(colour, in_spheres.top_priority_sphere(), t_range.end - t_prev)
}

fn refract_ray(
    ray: Vec3,
    normal: Vec3,
    from_sphere: Option<&Sphere>,
    to_sphere: Option<&Sphere>,
) -> Option<Vec3> {
    let c1 = from_sphere.map_or(1., |s| s.refractive);
    let c2 = to_sphere.map_or(1., |s| s.refractive);
    if c1 == 1. && c2 == 1. {
        return Some(ray);
    }
    let ri = c2 / c1;
    let ray = ray.as_unit();
    let d = -ray.dot(normal);
    let k = 1. - ri * ri * (1. - d * d);
    if k < 0. {
        None
    } else {
        let result = ri * ray + (ri * d - k.sqrt()) * normal;
        // if ri == 1. {
        //     debug_assert!((result - ray).norm() < EPSILON, "ray={:?} result={:?} normal={:?} d={} k={}", ray, result, normal, d, k);
        // }
        result.assert_ok();
        Some(result)
    }
}

fn reflect_ray(ray: Vec3, normal: Vec3) -> Vec3 {
    2. * normal.dot(ray) * normal - ray
}

fn pass_colour_thru_sphere(colour: Vec3, sphere: Option<&Sphere>, ray_len: f64) -> Vec3 {
    if let Some(s) = sphere {
        colour.pointwise_mul((s.transparent * s.colour_int).powf(ray_len))
    } else {
        colour
    }
}

fn canvas_to_viewport(x: u32, y: u32) -> Vec3 {
    Vec3(
        (x as f64 - CANVAS_WIDTH_F64 / 2.) / CANVAS_WIDTH_F64 * VIEWPORT_WIDTH,
        (y as f64 - CANVAS_HEIGHT_F64 / 2.) / CANVAS_HEIGHT_F64 * VIEWPORT_HEIGHT,
        VIEWPORT_DISTANCE,
    )
}

mod stats {

    #[cfg(debug_assertions)]
    #[derive(Debug, Default)]
    pub struct Stats {
        pub num_rays: u64,
        pub num_refract_rays: u64,
        pub num_reflect_rays: u64,
        pub num_double_adds: u64,
        pub num_double_removes: u64,
    }

    #[cfg(debug_assertions)]
    static mut STATS: Stats = Stats {
        num_rays: 0,
        num_reflect_rays: 0,
        num_refract_rays: 0,
        num_double_adds: 0,
        num_double_removes: 0,
    };

    #[cfg(debug_assertions)]
    pub fn get() -> &'static Stats {
        unsafe { &STATS }
    }

    pub fn record_ray() {
        #[cfg(debug_assertions)]
        unsafe {
            STATS.num_rays += 1;
        }
    }

    pub fn record_refract_ray() {
        #[cfg(debug_assertions)]
        unsafe {
            STATS.num_refract_rays += 1;
        }
    }

    pub fn record_reflect_ray() {
        #[cfg(debug_assertions)]
        unsafe {
            STATS.num_reflect_rays += 1;
        }
    }

    #[cfg(debug_assertions)]
    pub fn record_double_add() {
        unsafe {
            STATS.num_double_adds += 1;
        }
    }

    pub fn record_double_remove() {
        #[cfg(debug_assertions)]
        unsafe {
            STATS.num_double_removes += 1;
        }
    }
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
        colour_ext: parse_colour(v.find("colour_ext").unwrap()),
        colour_int: parse_opt_colour(v.find("colour_int"), Vec3::ONES),
        specular: parse_opt_f64(v.find("specular"), 0.),
        reflective: parse_opt_f64(v.find("reflective"), 0.),
        refractive: parse_opt_f64(v.find("refractive"), 1.),
        transparent: parse_opt_f64(v.find("transparent"), 0.),
        priority: parse_opt_u32(v.find("priority"), 0),
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

fn parse_opt_f64(vo: Option<&serde_hjson::Value>, default: f64) -> f64 {
    vo.map_or(default, |v| v.as_f64().unwrap())
}

fn parse_opt_u32(vo: Option<&serde_hjson::Value>, default: u32) -> u32 {
    vo.map_or(default, |v| v.as_u64().unwrap() as u32)
}

fn parse_opt_colour(vo: Option<&serde_hjson::Value>, default: Vec3) -> Vec3 {
    vo.map_or(default, |v| parse_colour(v))
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

fn raytrace_row<'a>(scene: &'a Scene, y: u32) -> Vec<image::Rgb<u8>> {
    let mut row = Vec::with_capacity(CANVAS_WIDTH as usize);
    for x in 0..CANVAS_WIDTH {
        let ray_dir = canvas_to_viewport(x, y);
        let colour = trace_ray(
            &scene,
            CAMERA,
            ray_dir,
            (1.)..f64::INFINITY,
            &SphereSet::new(),
            MAX_RECURSION,
        );
        row.push(colour.to_rgb_u8());
    }
    row
}

fn main() {
    let mut args = std::env::args().skip(1);
    let in_file = args.next().unwrap();
    let out_file = args.next().unwrap();
    let scene = parse_scene(&std::fs::read(in_file).unwrap());

    let rows = if cfg!(debug_assertions) {
        (0..CANVAS_HEIGHT)
            .into_iter()
            .map(|y| raytrace_row(&scene, y))
            .collect()
    } else {
        let mut rows = Vec::with_capacity(CANVAS_HEIGHT as usize);
        (0..CANVAS_HEIGHT)
            .into_par_iter()
            .map(|y| raytrace_row(&scene, y))
            .collect_into_vec(&mut rows);
        rows
    };

    let mut img = image::RgbImage::new(CANVAS_WIDTH, CANVAS_HEIGHT);

    for (y, row) in rows.iter().enumerate() {
        for (x, pixel) in row.iter().enumerate() {
            img.put_pixel(x as u32, CANVAS_HEIGHT - (y as u32) - 1, *pixel);
        }
    }

    img.save(out_file).unwrap();

    #[cfg(debug_assertions)]
    println!("{:?}", stats::get())
}
