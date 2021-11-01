// set the precision of the float values (necessary if using float)
#ifdef GL_FRAGMENT_PRECISION_HIGH
precision highp float;
#else
precision mediump float;
#endif
precision mediump int;

// flag for using soft shadows
#define SOFT_SHADOWS 0

// define number of soft shadow samples to take
#define SOFT_SAMPLING 3

// define constant parameters
// EPS is for the precision issue
#define INFINITY 1.0e+12
#define EPS 1.0e-3

// define maximum recursion depth for rays
#define MAX_RECURSION 8

// define constants for scene setting
#define MAX_LIGHTS 10

// define texture types
#define NONE 0
#define CHECKERBOARD 1
#define MYSPECIAL 2

// define material types
#define BASICMATERIAL 1
#define PHONGMATERIAL 2
#define LAMBERTMATERIAL 3

// define reflect types - how to bounce rays
#define NONEREFLECT 1
#define MIRRORREFLECT 2
#define GLASSREFLECT 3

struct Shape {
  int shapeType;
  vec3 v1;
  vec3 v2;
  float rad;
};

struct Material {
  int materialType;
  vec3 color;
  float shininess;
  vec3 specular;

  int materialReflectType;
  float reflectivity;
  float refractionRatio;
  int special;
};

struct Object {
  Shape shape;
  Material material;
};

struct Light {
  vec3 position;
  vec3 color;
  float intensity;
  float attenuate;
};

struct Ray {
  vec3 origin;
  vec3 direction;
};

struct Intersection {
  vec3 position;
  vec3 normal;
};

// uniform
uniform mat4 uMVMatrix;
uniform int frame;
uniform float height;
uniform float width;
uniform vec3 camera;
uniform int numObjects;
uniform int numLights;
uniform Light lights[MAX_LIGHTS];
uniform vec3 objectNorm;

// varying
varying vec2 v_position;

// find then position some distance along a ray
vec3 rayGetOffset(Ray ray, float dist) {
  return ray.origin + (dist * ray.direction);
}

// if a newly found intersection is closer than the best found so far, record
// the new intersection and return true; otherwise leave the best as it was and
// return false.
bool chooseCloserIntersection(float dist, inout float best_dist,
                              inout Intersection intersect,
                              inout Intersection best_intersect) {
  if (best_dist <= dist)
    return false;
  best_dist = dist;
  best_intersect.position = intersect.position;
  best_intersect.normal = intersect.normal;
  return true;
}

// put any general convenience functions you want up here
// ----------- STUDENT CODE BEGIN ------------
// ----------- Our reference solution uses 118 lines of code.
bool inBox(vec3 pmin, vec3 pmax, vec3 p){
  float xmin = pmin.x;
  float xmax = pmax.x;
  float ymin = pmin.y;
  float ymax = pmax.y;
  float zmin = pmin.z;
  float zmax = pmax.z;


  if (abs(p.x - xmin) < EPS || abs(p.x - xmax) < EPS){
    if (p.y > ymin && p.y < ymax){
      if (p.z > zmin && p.z < zmax) return true;
    }
  }

  if (abs(p.y - ymin) < EPS || abs(p.y - ymax) < EPS){
    if (p.x > xmin && p.x < xmax){
      if (p.z > zmin && p.z < zmax) return true;
    }
  }

  if (abs(p.z - zmin) < EPS || abs(p.z - zmax) < EPS){
    if (p.y > ymin && p.y < ymax){
      if (p.x > xmin && p.x < xmax) return true;
    }
  }
 
  return false;
}

float interpolate(float a0, float a1, float w){
  return (a1-a0) * w + a0;
}

vec2 randomGradient(float ix, float iy){

  float random = 2920382940.0 * sin(ix * 21402.0 + iy * 171324.0 + 3213214.0) * cos(ix * 19032134.0 + iy * 21324.0 + 432149.0) ;
  return vec2(cos(random), sin(random));
}

float dotGridGradient(int ix, int iy, float x, float y){
  vec2 gradient = randomGradient(float(ix), float(iy));

  float dx = x - float(ix);
  float dy = y - float(iy);

  return (dx*gradient.x + dy * gradient.y);
}

float perlin(float x, float y){
  int x0 = int(x);
  int x1 = x0 + 1;
  int y0 = int(y);
  int y1 = y0 + 1;

  float sx = x - float(x0);
  float sy = y - float(y0);

  float n0 = dotGridGradient(x0, y0, x, y);
  float n1 = dotGridGradient(x1, y0, x, y);
  float ix0 = interpolate(n0, n1, sx);


  n0 = dotGridGradient(x0, y1, x, y);
  n1 = dotGridGradient(x1, y1, x, y);
  float ix1 = interpolate(n0, n1, sx);
  float value = interpolate(ix0, ix1,sy);

  return value;
}

// ----------- STUDENT CODE END ------------

// forward declaration
float rayIntersectScene(Ray ray, out Material out_mat,
                        out Intersection out_intersect);

// Plane
// this function can be used for plane, triangle, and box
float findIntersectionWithPlane(Ray ray, vec3 norm, float dist,
                                out Intersection intersect) {
  float a = dot(ray.direction, norm);
  float b = dot(ray.origin, norm) - dist;

  if (a < EPS && a > -EPS)
    return INFINITY;

  float len = -b / a;
  if (len < EPS)
    return INFINITY;

  intersect.position = rayGetOffset(ray, len);
  intersect.normal = norm;
  return len;
}

// Triangle
float findIntersectionWithTriangle(Ray ray, vec3 t1, vec3 t2, vec3 t3,
                                   out Intersection intersect) {
  // ----------- STUDENT CODE BEGIN ------------
  // ----------- Our reference solution uses 28 lines of code.
  vec3 norm = normalize(cross(t2-t1,t3-t1));


  //if (dot(ray.direction, norm) < EPS) return INFINITY;
  
  float dist =  dot(norm, t1);
  Intersection dummy;
  float len = findIntersectionWithPlane(ray, norm, dist, dummy);
  if (abs(INFINITY - len) < EPS) return INFINITY;



  vec3 v1 = normalize(t1 - dummy.position);
  vec3 v2 = normalize(t2 - dummy.position);
  vec3 n1 = normalize(cross(v2, v1));
  if(dot(ray.direction, n1) < EPS) return INFINITY;

  v2 = normalize(t2 - dummy.position);
  vec3 v3 = normalize(t3 - dummy.position);
  vec3 n2 = normalize(cross(v3, v2));
  if(dot(ray.direction, n2) < EPS) return INFINITY;

  v3 = normalize(t3 - dummy.position);
  v1 = normalize(t1 - dummy.position);
  vec3 n3 = normalize(cross(v1, v3));
  if(dot(ray.direction, n3) < EPS) return INFINITY;
  
  intersect.normal = norm;
  intersect.position = dummy.position;


  return len;
  // ----------- STUDENT CODE END ------------
}

// Sphere
float findIntersectionWithSphere(Ray ray, vec3 center, float radius,
                                 out Intersection intersect) {
  // ----------- STUDENT CODE BEGIN ------------
  // ----------- Our reference solution uses 25 lines of code.
  vec3 L = center - ray.origin;
  float tca = dot(L, ray.direction);
  if (tca < EPS) return INFINITY;

  float d2 = dot(L, L) - tca*tca;
  if (d2 > radius*radius) return INFINITY;

  float thc = sqrt(radius*radius - d2);
  
  float t1 = tca + thc;
  float t2 = tca - thc;
  float dist = 0.1;
  if(t1 < t2 && t1 > EPS) {
    intersect.position = rayGetOffset(ray, t1);
    dist = t1;
  }
  else if (t2 < t1 && t2 > EPS){
    intersect.position = rayGetOffset(ray, t2);
    dist = t2;
  }
  else if (t1 > EPS){
    intersect.position = rayGetOffset(ray, t1);
    dist = t1;
  }
  else if(t2 > EPS){
    intersect.position = rayGetOffset(ray, t2);
    dist = t2;
  }
  

  else return INFINITY;

  intersect.normal = normalize(intersect.position - center);

  return dist;
  // ----------- STUDENT CODE END ------------
}


// Box
float findIntersectionWithBox(Ray ray, vec3 pmin, vec3 pmax,
                              out Intersection out_intersect) {
  // ----------- STUDENT CODE BEGIN ------------
  // pmin and pmax represent two bounding points of the box
  // pmin stores [xmin, ymin, zmin] and pmax stores [xmax, ymax, zmax]
  // ----------- Our reference solution uses 44 lines of code.
  // currently reports no intersection
  Intersection dummy;
  float best_len = INFINITY;
  Intersection best_intersect;
  vec3 bottomleft; 
  vec3 topright;
  vec3 bottomright;
  vec3 norm;
  float d;
  float len;
  bool box;
  bool closer;
  
  // front
  bottomleft = pmin;
  topright = vec3(pmax.x, pmin.y, pmax.z);
  bottomright = vec3(pmax.x, pmin.y, pmin.z);

  norm = normalize(cross(topright - bottomright, bottomleft - bottomright));
  d = dot(bottomleft, norm);
  
  len = findIntersectionWithPlane(ray, norm, d, dummy);


  box = inBox(pmin, pmax, dummy.position);

  if(box && len > EPS) closer = chooseCloserIntersection(len, best_len, dummy, best_intersect);



  // right
  bottomleft = vec3(pmax.x, pmin.y, pmin.z);
  topright = pmax;
  bottomright = vec3(pmax.x, pmax.y, pmin.z);

  norm = normalize(cross(topright - bottomright, bottomleft - bottomright));
  d = dot(bottomleft, norm);
  
  len = findIntersectionWithPlane(ray, norm, d, dummy);


  box = inBox(pmin, pmax, dummy.position);

  if(box && len > EPS) closer = chooseCloserIntersection(len, best_len, dummy, best_intersect);
   
   
  // left
  bottomleft = vec3(pmin.x, pmax.y, pmin.z);
  topright = vec3(pmin.x, pmin.y, pmax.z);
  bottomright = pmin;

  norm = normalize(cross(topright - bottomright, bottomleft - bottomright));
  d = dot(bottomleft, norm);
  
  len = findIntersectionWithPlane(ray, norm, d, dummy);


  box = inBox(pmin, pmax, dummy.position);

  if(box && len > EPS) closer = chooseCloserIntersection(len, best_len, dummy, best_intersect);


  // top
  bottomleft = vec3(pmin.x, pmin.y, pmax.z);
  topright = pmax;
  bottomright = vec3(pmax.x, pmin.y, pmax.z);

  norm = normalize(cross(topright - bottomright, bottomleft - bottomright));
  d = dot(bottomleft, norm);
  
  len = findIntersectionWithPlane(ray, norm, d, dummy);


  box = inBox(pmin, pmax, dummy.position);

  if(box && len > EPS) closer = chooseCloserIntersection(len, best_len, dummy, best_intersect);


   // back
  bottomleft = vec3(pmax.x, pmax.y, pmin.z);
  topright = vec3(pmin.x, pmax.y, pmax.z);
  bottomright = vec3(pmin.x, pmax.y, pmin.z);

  norm = normalize(cross(topright - bottomright, bottomleft - bottomright));
  d = dot(bottomleft, norm);
  
  len = findIntersectionWithPlane(ray, norm, d, dummy);


  box = inBox(pmin, pmax, dummy.position);

  if(box && len > EPS) closer = chooseCloserIntersection(len, best_len, dummy, best_intersect);


  // bottom
  bottomleft = vec3(pmin.x, pmax.y, pmin.z);
  topright = vec3(pmax.x, pmin.y, pmin.z);
  bottomright = vec3(pmax.x, pmax.y, pmin.z);

  norm = normalize(cross(topright - bottomright, bottomleft - bottomright));
  d = dot(bottomleft, norm);
  
  len = findIntersectionWithPlane(ray, norm, d, dummy);


  box = inBox(pmin, pmax, dummy.position);

  if(box && len > EPS) closer = chooseCloserIntersection(len, best_len, dummy, best_intersect);


  out_intersect.normal = best_intersect.normal;
  out_intersect.position = best_intersect.position;
  
  return best_len;
  
  // ----------- STUDENT CODE END ------------
}

// Cylinder
float getIntersectOpenCylinder(Ray ray, vec3 center, vec3 axis, float len,
                               float rad, out Intersection intersect) {
  // ----------- STUDENT CODE BEGIN ------------
  // ----------- Our reference solution uses 33 lines of code.
  vec3 q;
  vec3 p1 = center;
  vec3 p2 = center + normalize(axis) * len;
  vec3 vd = ray.origin - center;
  vec3 va = normalize(axis);
  vec3 vr = ray.direction;
  vec3 xy_norm = vec3(0,1,0);

  float phi = dot(vd, va);
  float theta = dot(vr, va);

  float a = pow(length(vr - theta * va),2.0);
  float b = dot(2.0 * (vr - theta*va), (vd - phi * va));
  float c = pow(length(vd - phi * va),2.0) - pow(rad, 2.0); 

  float d1 = (-b + sqrt(pow(b,2.0) - 4.0 * a * c))/(2.0*a);
  float d2 = (-b - sqrt(pow(b,2.0) - 4.0 * a * c))/(2.0*a);

  if (d1 < EPS && d2 < EPS) return INFINITY;
  else if (d1 < EPS) {
    q = rayGetOffset(ray, d2);
    if(dot(va, q - p1) > EPS && dot(va, q - p2) < EPS){
      intersect.position = q;
      intersect.normal.xz = normalize(q.xz - center.xz);
      intersect.normal.y = 0.0;

      return d2;
    }
  }
  else if (d2 < EPS) {
    q = rayGetOffset(ray, d1);
    if(dot(va, q - p1) > EPS && dot(va, q - p2) < EPS){
      intersect.position = q;
      intersect.normal.xz = normalize(q.xz - center.xz);
      intersect.normal.y = 0.0;
      return d1;
    }
  }
  else if (d2 > d1){
    q = rayGetOffset(ray, d1);
    if(dot(va, q - p1) > EPS && dot(va, q - p2) < EPS){
      intersect.position = q;
      intersect.normal.xz = normalize(q.xz - center.xz);
      intersect.normal.y = 0.0;
      return d1;
    }
  }
  else if (d1 > d2){
    q = rayGetOffset(ray, d2);
    if(dot(va, q - p1) > EPS && dot(va, q - p2) < EPS){
      intersect.position = q;
      intersect.normal.xz = normalize(q.xz - center.xz);
      intersect.normal.y = 0.0;      
      return d2;
    }
  }


  // currently reports no intersection
  return INFINITY;
  // ----------- STUDENT CODE END ------------
}

float getIntersectDisc(Ray ray, vec3 center, vec3 norm, float rad,
                       out Intersection intersect) {
  // ----------- STUDENT CODE BEGIN ------------
  // ----------- Our reference solution uses 18 lines of code.
  float dist =  dot(norm, center);

  float len = findIntersectionWithPlane(ray, norm, dist, intersect);

  if (pow(length(intersect.position - center), 2.0) < pow(rad,2.0)){
    return len;
  }
  else return INFINITY;
  // ----------- STUDENT CODE END ------------
}

float findIntersectionWithCylinder(Ray ray, vec3 center, vec3 apex,
                                   float radius,
                                   out Intersection out_intersect) {
  vec3 axis = apex - center;
  float len = length(axis);
  axis = normalize(axis);

  Intersection intersect;
  float best_dist = INFINITY;
  float dist;

  // -- infinite cylinder
  dist = getIntersectOpenCylinder(ray, center, axis, len, radius, intersect);
  chooseCloserIntersection(dist, best_dist, intersect, out_intersect);

  // -- two caps
  dist = getIntersectDisc(ray, center, -axis, radius, intersect);
  chooseCloserIntersection(dist, best_dist, intersect, out_intersect);
  dist = getIntersectDisc(ray, apex, axis, radius, intersect);
  chooseCloserIntersection(dist, best_dist, intersect, out_intersect);
  return best_dist;
}

// Cone
float getIntersectOpenCone(Ray ray, vec3 apex, vec3 axis, float len,
                           float rad, out Intersection intersect) {
  // ----------- STUDENT CODE BEGIN ------------
  // ----------- Our reference solution uses 45 lines of code.
  // currently reports no intersection
  vec3 pr = ray.origin;
  vec3 pa = apex;
  vec3 vr = ray.direction;
  vec3 va = axis; // maybe
  vec3 pc = pa + axis * len;

  vec3 vd = pr - pa;
  float phi = dot(vd, va);
  float theta = dot(vr, va);

  vec3 q;

  float hyp = sqrt(pow(rad,2.0) + pow(len, 2.0));
  float cosa = len/hyp;
  float sina = rad/hyp;

  float a = pow(length(vr - theta*va), 2.0) * pow(cosa,2.0) - pow(theta,2.0)*pow(sina,2.0);
  float b = 2.0*((dot(vr-theta*va, vd - phi*va))*pow(cosa, 2.0) - theta * phi * pow(sina,2.0));
  float c = pow(length(vd-phi*va),2.0) * pow(cosa, 2.0) - pow(phi,2.0) * pow(sina, 2.0);
  
  float d1 = (-b + sqrt(pow(b,2.0) - 4.0 * a * c))/(2.0*a);
  float d2 = (-b - sqrt(pow(b,2.0) - 4.0 * a * c))/(2.0*a);


  if (d1 < EPS && d2 < EPS) return INFINITY;
  else if (d1 < EPS) {
    q = rayGetOffset(ray, d2);
    if(dot(va, q - pa) > EPS && dot(va, q - pc) < EPS){
      intersect.position = q;
      intersect.normal.xz = normalize(q.xz - apex.xz);
      intersect.normal.y = rad/len;

      return d2;
    }
  }
  else if (d2 < EPS) {
    q = rayGetOffset(ray, d1);
    if(dot(va, q - pa) > EPS && dot(va, q - pc) < EPS){
      intersect.position = q;
      intersect.normal.xz = normalize(q.xz - apex.xz);
      intersect.normal.y = rad/len;

      return d1;
    }
  }
  else if (d2 > d1){
    q = rayGetOffset(ray, d1);
    if(dot(va, q - pa) > EPS && dot(va, q - pc) < EPS){
      intersect.position = q;
      intersect.normal.xz = normalize(q.xz - apex.xz);
      intersect.normal.y = rad/len;

      return d1;
    }
  }
  else if (d1 > d2){
    q = rayGetOffset(ray, d2);
    if(dot(va, q - pa) > EPS && dot(va, q - pc) < EPS){
      intersect.position = q;
      intersect.normal.xz = normalize(q.xz - apex.xz);
      intersect.normal.y = rad/len;

      return d2;
    }
  }



  return INFINITY;
  // ----------- STUDENT CODE END ------------
}

float findIntersectionWithCone(Ray ray, vec3 center, vec3 apex, float radius,
                               out Intersection out_intersect) {
  vec3 axis = center - apex;
  float len = length(axis);
  axis = normalize(axis);

  // -- infinite cone
  Intersection intersect;
  float best_dist = INFINITY;
  float dist;

  // -- infinite cone
  dist = getIntersectOpenCone(ray, apex, axis, len, radius, intersect);
  chooseCloserIntersection(dist, best_dist, intersect, out_intersect);

  // -- caps
  dist = getIntersectDisc(ray, center, axis, radius, intersect);
  chooseCloserIntersection(dist, best_dist, intersect, out_intersect);

  return best_dist;
}

vec3 calculateSpecialDiffuseColor(Material mat, vec3 posIntersection,
                                  vec3 normalVector) {
  // ----------- STUDENT CODE BEGIN ------------
  if (mat.special == CHECKERBOARD) {
    // ----------- Our reference solution uses 7 lines of code.
    vec3 white = vec3(1,1,1);
    vec3 black = vec3(0,0,0);
    float size = 10.0;

    float quant = floor((posIntersection.x + EPS)/size) + floor((posIntersection.y + EPS)/size) + floor((posIntersection.z + EPS)/size);
    quant = mod(quant, 2.0);
    if (quant < EPS) mat.color = white;
    else mat.color = black;

  } else if (mat.special == MYSPECIAL) {
    // ----------- Our reference solution uses 5 lines of code.
    mat.color.x = mat.color.x + perlin(posIntersection.x, posIntersection.y);
    mat.color.y = mat.color.y + perlin(posIntersection.x, posIntersection.y);
    mat.color.z = mat.color.z + perlin(posIntersection.x, posIntersection.y);


  }

  // If not a special material, just return material color.
  return mat.color;
  // ----------- STUDENT CODE END ------------
}

vec3 calculateDiffuseColor(Material mat, vec3 posIntersection,
                           vec3 normalVector) {
  // Special colors
  if (mat.special != NONE) {
    return calculateSpecialDiffuseColor(mat, posIntersection, normalVector);
  }
  return vec3(mat.color);
}

// check if position pos in in shadow with respect to a particular light.
// lightVec is the vector from that position to that light -- it is not
// normalized, so its length is the distance from the position to the light
bool pointInShadow(vec3 pos, vec3 lightVec) {
  // ----------- STUDENT CODE BEGIN ------------
  // ----------- Our reference solution uses 15 lines of code.
  Ray r;
  Material mat;
  Intersection inter;

  vec3 zero = vec3(0,0,0);

  r.direction = zero - normalize(lightVec);
  r.origin = pos + lightVec;

  float dist =  rayIntersectScene(r, mat, inter);

  if (abs(dist-length(lightVec)) < EPS) return false;

  return true;
  // ----------- STUDENT CODE END ------------
}

// use random sampling to compute a ratio that represents the
// fractional contribution of the light to the position pos.
// lightVec is the vector from that position to that light -- it is not
// normalized, so its length is the distance from the position to the light
float softShadowRatio(vec3 pos, vec3 lightVec) {
  // ----------- STUDENT CODE BEGIN ------------
  // ----------- Our reference solution uses 19 lines of code.
  return 0.0;
  // ----------- STUDENT CODE END ------------
}

vec3 getLightContribution(Light light, Material mat, vec3 posIntersection,
                          vec3 normalVector, vec3 eyeVector, bool phongOnly,
                          vec3 diffuseColor) {
  vec3 lightVector = light.position - posIntersection;


  float ratio = 1.0; // default to 1.0 for hard shadows
  if (SOFT_SHADOWS == 1) {
    // if using soft shadows, call softShadowRatio to determine
    // fractional light contribution
    ratio = softShadowRatio(posIntersection, lightVector);
  }
  else {
    // check if point is in shadow with light vector
    if (pointInShadow(posIntersection, lightVector)) {
      return vec3(0.0, 0.0, 0.0);
    }
  }

  // Slight optimization for soft shadows
  if (ratio < EPS) {
    return vec3(0.0, 0.0, 0.0);
  }


  // normalize the light vector for the computations below
  float distToLight = length(lightVector);
  lightVector /= distToLight;

  if (mat.materialType == PHONGMATERIAL ||
      mat.materialType == LAMBERTMATERIAL) {
    vec3 contribution = vec3(0.0, 0.0, 0.0);

    // get light attenuation
    float attenuation = light.attenuate * distToLight;
    float diffuseIntensity =
        max(0.0, dot(normalVector, lightVector)) * light.intensity;

    // glass and mirror objects have specular highlights but no diffuse lighting
    if (!phongOnly) {
      contribution += 
          diffuseColor * diffuseIntensity * light.color / attenuation;
    }

    if (mat.materialType == PHONGMATERIAL) {
      // Start with just black by default (i.e. no Phong term contribution)
      vec3 phongTerm = vec3(0.0, 0.0, 0.0);
      // ----------- STUDENT CODE BEGIN ------------
      // ----------- Our reference solution uses 4 lines of code.
      vec3 r =  reflect(-1.0*lightVector, normalVector);
      phongTerm +=  mat.specular * pow(max(0.0, dot(normalize(eyeVector), normalize(r))), mat.shininess)*light.intensity/attenuation;
      // ----------- STUDENT CODE END ------------
      contribution += phongTerm;
    }

    return ratio * contribution;
  } else {
    return ratio * diffuseColor;
  }
}

vec3 calculateColor(Material mat, vec3 posIntersection, vec3 normalVector,
                    vec3 eyeVector, bool phongOnly) {
  // The diffuse color of the material at the point of intersection
  // Needed to compute the color when accounting for the lights in the scene
  vec3 diffuseColor = calculateDiffuseColor(mat, posIntersection, normalVector);

  // color defaults to black when there are no lights
  vec3 outputColor = vec3(0.0, 0.0, 0.0);

  // Loop over the MAX_LIGHTS different lights, taking care not to exceed
  // numLights (GLSL restriction), and accumulate each light's contribution
  // to the point of intersection in the scene.
  // ----------- STUDENT CODE BEGIN ------------
  // ----------- Our reference solution uses 9 lines of code.
  for (int i = 0; i < MAX_LIGHTS; i++){
    if (i >= numLights) break;
    outputColor += getLightContribution(lights[i], mat, posIntersection, normalVector, eyeVector, phongOnly, diffuseColor);
  
  }
  // Return diffuseColor by default, so you can see something for now.
  return outputColor;
  // ----------- STUDENT CODE END ------------
}

// find reflection or refraction direction (depending on material type)
vec3 calcReflectionVector(Material material, vec3 direction, vec3 normalVector,
                          bool isInsideObj) {
  if (material.materialReflectType == MIRRORREFLECT) {
    return reflect(direction, normalVector);
  }
  // If it's not mirror, then it is a refractive material like glass.
  // Compute the refraction direction.
  // See lecture 13 slide (lighting) on Snell's law.
  // The eta below is eta_i/eta_r.
  // ----------- STUDENT CODE BEGIN ------------
  float eta =
      (isInsideObj) ? 1.0 / material.refractionRatio : material.refractionRatio;
  // ----------- Our reference solution uses 5 lines of code.
  // Return mirror direction by default, so you can see something for now.
  float costheta1 = dot(normalVector*-1.0, direction);
  float costheta2 = sqrt(1.0 - eta*eta*(1.0 -costheta1*costheta1));
  vec3 outvector = direction*eta + normalVector * (eta*costheta1 - costheta2);
  float sintheta2 = eta * sqrt(1.0 - costheta1*costheta1);

  if(sintheta2 >= 1.0 && isInsideObj) return vec3(0,0,0); 
  else if (sintheta2 >= 1.0) return reflect(direction, normalVector);

  return outvector;
  // ----------- STUDENT CODE END ------------
}

vec3 traceRay(Ray ray) {
  // Accumulate the final color from tracing this ray into resColor.
  vec3 resColor = vec3(0.0, 0.0, 0.0);

  // Accumulate a weight from tracing this ray through different materials
  // based on their BRDFs. Initially all 1.0s (i.e. scales the initial ray's
  // RGB color by 1.0 across all color channels). This captures the BRDFs
  // of the materials intersected by the ray's journey through the scene.
  vec3 resWeight = vec3(1.0, 1.0, 1.0);

  // Flag for whether the ray is currently inside of an object.
  bool isInsideObj = false;

  // Iteratively trace the ray through the scene up to MAX_RECURSION bounces.
  for (int depth = 0; depth < MAX_RECURSION; depth++) {
    // Fire the ray into the scene and find an intersection, if one exists.
    //
    // To do so, trace the ray using the rayIntersectScene function, which
    // also accepts a Material struct and an Intersection struct to store
    // information about the point of intersection. The function returns
    // a distance of how far the ray travelled before it intersected an object.
    //
    // Then, check whether or not the ray actually intersected with the scene.
    // A ray does not intersect the scene if it intersects at a distance
    // "equal to zero" or far beyond the bounds of the scene. If so, break
    // the loop and do not trace the ray any further.
    // (Hint: You should probably use EPS and INFINITY.)
    // ----------- STUDENT CODE BEGIN ------------
    Material hitMaterial;
    Intersection intersect;
    // ----------- Our reference solution uses 4 lines of code.
    float dist = rayIntersectScene(ray, hitMaterial, intersect);
    if (dist < EPS) break;
    if (dist > INFINITY) break;

    // ----------- STUDENT CODE END ------------

    // Compute the vector from the ray towards the intersection.
    vec3 posIntersection = intersect.position;
    vec3 normalVector    = intersect.normal;

    vec3 eyeVector = normalize(ray.origin - posIntersection);

    // Determine whether we are inside an object using the dot product
    // with the intersection's normal vector
    if (dot(eyeVector, normalVector) < 0.0) {
        normalVector = -normalVector;
        isInsideObj = true;
    } else {
        isInsideObj = false;
    }

    // Material is reflective if it is either mirror or glass in this assignment
    bool reflective = (hitMaterial.materialReflectType == MIRRORREFLECT ||
                       hitMaterial.materialReflectType == GLASSREFLECT);

    // Compute the color at the intersection point based on its material
    // and the lighting in the scene
    vec3 outputColor = calculateColor(hitMaterial, posIntersection,
      normalVector, eyeVector, reflective);

    // A material has a reflection type (as seen above) and a reflectivity
    // attribute. A reflectivity "equal to zero" indicates that the material
    // is neither reflective nor refractive.

    // If a material is neither reflective nor refractive...
    // (1) Scale the output color by the current weight and add it into
    //     the accumulated color.
    // (2) Then break the for loop (i.e. do not trace the ray any further).
    // ----------- STUDENT CODE BEGIN ------------
    // ----------- Our reference solution uses 4 lines of code.
    if (hitMaterial.reflectivity < EPS) {
      resColor += outputColor * resWeight;
      break;
    }
    // ----------- STUDENT CODE END ------------

    // If the material is reflective or refractive...
    // (1) Use calcReflectionVector to compute the direction of the next
    //     bounce of this ray.
    // (2) Update the ray object with the next starting position and
    //     direction to prepare for the next bounce. You should modify the
    //     ray's origin and direction attributes. Be sure to normalize the
    //     direction vector.
    // (3) Scale the output color by the current weight and add it into
    //     the accumulated color.
    // (4) Update the current weight using the material's reflectivity
    //     so that it is the appropriate weight for the next ray's color.
    // ----------- STUDENT CODE BEGIN ------------
    // ----------- Our reference solution uses 8 lines of code.
    ray.direction  = normalize(calcReflectionVector(hitMaterial, ray.direction, normalVector, isInsideObj));
    ray.origin = posIntersection;
    resColor += outputColor * resWeight;
    resWeight = resWeight * hitMaterial.reflectivity;
    // ----------- STUDENT CODE END ------------
  }

  return resColor;
}

void main() {
  float cameraFOV = 0.8;
  vec3 direction = vec3(v_position.x * cameraFOV * width / height,
                        v_position.y * cameraFOV, 1.0);

  Ray ray;
  ray.origin = vec3(uMVMatrix * vec4(camera, 1.0));
  ray.direction = normalize(vec3(uMVMatrix * vec4(direction, 0.0)));

  // trace the ray for this pixel
  vec3 res = traceRay(ray);

  // paint the resulting color into this pixel
  gl_FragColor = vec4(res.x, res.y, res.z, 1.0);
}
