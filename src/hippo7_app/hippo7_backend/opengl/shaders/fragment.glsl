#define GLSLIFY 1
vec4 blur5(sampler2D image, vec2 uv, vec2 resolution, vec2 direction) {
  vec4 color = vec4(0.0);
  vec2 off1 = vec2(1.3333333333333333) * direction;
  color += texture2D(image, uv) * 0.29411764705882354;
  color += texture2D(image, uv + (off1 / resolution)) * 0.35294117647058826;
  color += texture2D(image, uv - (off1 / resolution)) * 0.35294117647058826;
  return color; 
}

const mat3 X_COMPONENT_MATRIX_529295689 = mat3(
  1., 0., -1.,
  2., 0., -2.,
  1., 0., -1.
);
const mat3 Y_COMPONENT_MATRIX_529295689 = mat3(
  1., 2., 1.,
  0., 0., 0.,
  -1., -2., -1.
);

/**
 * 3x3 Matrix convolution
 */
float convoluteMatrices(mat3 A, mat3 B) {
  return dot(A[0], B[0]) + dot(A[1], B[1]) + dot(A[2], B[2]);
}

/**
 * Get the color of a texture after
 * a Guassian blur with a radius of 5 pixels
 */
vec3 getBlurredTextureColor(
  sampler2D textureSampler,
  vec2 textureCoord,
  vec2 resolution
) {
  return blur5(
    textureSampler,
    textureCoord,
    resolution,
    normalize(textureCoord - vec2(0.5))).xyz;
}

/**
 * Get the intensity of the color on a
 * texture after a guassian blur is applied
 */
float getTextureIntensity(
  sampler2D textureSampler,
  vec2 textureCoord,
  vec2 resolution
) {
  vec3 color = getBlurredTextureColor(textureSampler, textureCoord, resolution);
  return pow(length(clamp(color, vec3(0.), vec3(1.))), 2.) / 3.;
}

/**
 * Get the gradient of the textures intensity
 * as a function of the texture coordinate
 */
vec2 getTextureIntensityGradient(
  sampler2D textureSampler,
  vec2 textureCoord,
  vec2 resolution
) {
  vec2 gradientStep = vec2(1.) / resolution;

  mat3 imgMat = mat3(0.);

  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      vec2 ds = vec2(
        -gradientStep.x + (float(i) * gradientStep.x),
        -gradientStep.y + (float(j) * gradientStep.y));
      imgMat[i][j] = getTextureIntensity(
        textureSampler, clamp(textureCoord + ds, vec2(0.), vec2(1.)), resolution);
    }
  }

  float gradX = convoluteMatrices(X_COMPONENT_MATRIX_529295689, imgMat);
  float gradY = convoluteMatrices(Y_COMPONENT_MATRIX_529295689, imgMat);

  return vec2(gradX, gradY);
}

float luma(vec3 color) {
  return dot(color, vec3(0.299, 0.587, 0.114));
}

float luma(vec4 color) {
  return dot(color.rgb, vec3(0.299, 0.587, 0.114));
}

vec3 crosshatch(vec3 texColor, float t1, float t2, float t3, float t4) {
  float lum = luma(texColor);
  vec3 color = vec3(1.0);
  if (lum < t1) {
      if (mod(gl_FragCoord.x + gl_FragCoord.y, 10.0) == 0.0) {
          color = vec3(0.0);
      }
  }
  if (lum < t2) {
      if (mod(gl_FragCoord.x - gl_FragCoord.y, 10.0) == 0.0) {
          color = vec3(0.0);
      }
  }
  if (lum < t3) {
      if (mod(gl_FragCoord.x + gl_FragCoord.y - 5.0, 10.0) == 0.0) {
          color = vec3(0.0);
      }
  }
  if (lum < t4) {
      if (mod(gl_FragCoord.x - gl_FragCoord.y - 5.0, 10.0) == 0.0) {
          color = vec3(0.0);
      }
  }
  return color;
}

vec3 crosshatch(vec3 texColor) {
  return crosshatch(texColor, 1.0, 0.75, 0.5, 0.3);
}

//
// Description : Array and textureless GLSL 2D simplex noise function.
//      Author : Ian McEwan, Ashima Arts.
//  Maintainer : ijm
//     Lastmod : 20110822 (ijm)
//     License : Copyright (C) 2011 Ashima Arts. All rights reserved.
//               Distributed under the MIT License. See LICENSE file.
//               https://github.com/ashima/webgl-noise
//

vec3 mod289(vec3 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec2 mod289(vec2 x) {
  return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec3 permute(vec3 x) {
  return mod289(((x*34.0)+1.0)*x);
}

float snoise(vec2 v)
  {
  const vec4 C = vec4(0.211324865405187,  // (3.0-sqrt(3.0))/6.0
                      0.366025403784439,  // 0.5*(sqrt(3.0)-1.0)
                     -0.577350269189626,  // -1.0 + 2.0 * C.x
                      0.024390243902439); // 1.0 / 41.0
// First corner
  vec2 i  = floor(v + dot(v, C.yy) );
  vec2 x0 = v -   i + dot(i, C.xx);

// Other corners
  vec2 i1;
  //i1.x = step( x0.y, x0.x ); // x0.x > x0.y ? 1.0 : 0.0
  //i1.y = 1.0 - i1.x;
  i1 = (x0.x > x0.y) ? vec2(1.0, 0.0) : vec2(0.0, 1.0);
  // x0 = x0 - 0.0 + 0.0 * C.xx ;
  // x1 = x0 - i1 + 1.0 * C.xx ;
  // x2 = x0 - 1.0 + 2.0 * C.xx ;
  vec4 x12 = x0.xyxy + C.xxzz;
  x12.xy -= i1;

// Permutations
  i = mod289(i); // Avoid truncation effects in permutation
  vec3 p = permute( permute( i.y + vec3(0.0, i1.y, 1.0 ))
    + i.x + vec3(0.0, i1.x, 1.0 ));

  vec3 m = max(0.5 - vec3(dot(x0,x0), dot(x12.xy,x12.xy), dot(x12.zw,x12.zw)), 0.0);
  m = m*m ;
  m = m*m ;

// Gradients: 41 points uniformly over a line, mapped onto a diamond.
// The ring size 17*17 = 289 is close to a multiple of 41 (41*7 = 287)

  vec3 x = 2.0 * fract(p * C.www) - 1.0;
  vec3 h = abs(x) - 0.5;
  vec3 ox = floor(x + 0.5);
  vec3 a0 = x - ox;

// Normalise gradients implicitly by scaling m
// Approximation of: m *= inversesqrt( a0*a0 + h*h );
  m *= 1.79284291400159 - 0.85373472095314 * ( a0*a0 + h*h );

// Compute final noise value at P
  vec3 g;
  g.x  = a0.x  * x0.x  + h.x  * x0.y;
  g.yz = a0.yz * x12.xz + h.yz * x12.yw;
  return 130.0 * dot(m, g);
}

float aastep(float threshold, float value) {
  #ifdef GL_OES_standard_derivatives
    float afwidth = length(vec2(dFdx(value), dFdy(value))) * 0.70710678118654757;
    return smoothstep(threshold-afwidth, threshold+afwidth, value);
  #else
    return step(threshold, value);
  #endif  
}

vec3 halftone(vec3 texcolor, vec2 st, float frequency) {
  float n = 0.1*snoise(st*200.0); // Fractal noise
  n += 0.05*snoise(st*400.0);
  n += 0.025*snoise(st*800.0);
  vec3 white = vec3(n*0.2 + 0.97);
  vec3 black = vec3(n + 0.1);

  // Perform a rough RGB-to-CMYK conversion
  vec4 cmyk;
  cmyk.xyz = 1.0 - texcolor;
  cmyk.w = min(cmyk.x, min(cmyk.y, cmyk.z)); // Create K
  cmyk.xyz -= cmyk.w; // Subtract K equivalent from CMY

  // Distance to nearest point in a grid of
  // (frequency x frequency) points over the unit square
  vec2 Kst = frequency*mat2(0.707, -0.707, 0.707, 0.707)*st;
  vec2 Kuv = 2.0*fract(Kst)-1.0;
  float k = aastep(0.0, sqrt(cmyk.w)-length(Kuv)+n);
  vec2 Cst = frequency*mat2(0.966, -0.259, 0.259, 0.966)*st;
  vec2 Cuv = 2.0*fract(Cst)-1.0;
  float c = aastep(0.0, sqrt(cmyk.x)-length(Cuv)+n);
  vec2 Mst = frequency*mat2(0.966, 0.259, -0.259, 0.966)*st;
  vec2 Muv = 2.0*fract(Mst)-1.0;
  float m = aastep(0.0, sqrt(cmyk.y)-length(Muv)+n);
  vec2 Yst = frequency*st; // 0 deg
  vec2 Yuv = 2.0*fract(Yst)-1.0;
  float y = aastep(0.0, sqrt(cmyk.z)-length(Yuv)+n);

  vec3 rgbscreen = 1.0 - 0.9*vec3(c,m,y) + n;
  return mix(rgbscreen, black, 0.85*k + 0.3*n);
}

vec3 halftone(vec3 texcolor, vec2 st) {
  return halftone(texcolor, st, 30.0);
}

uniform sampler2D tex;

uniform vec2 u_resolution;
uniform int apply_sobel; //Bool somehow doesnt work :/
uniform int mirror_x;
uniform int mirror_y;

uniform int is_crosshatch;
uniform int is_inverted;
uniform float halftone_resolution;

uniform vec2 repeat;
uniform vec2 texture_scale;
uniform float brightness;
uniform float transparency;

varying vec2 v_texcoord;

float mirror_repeat(float x, float repeat, int mirror){
	if (repeat>1.0){
		repeat = 1./repeat;
		x = mod(x,repeat)/repeat;
	} else {
		repeat= 1.0;
	}
	if (mirror != 1) return x;
	return 1.0-abs(2.0*x-1.0);
}

void main() {
	vec2 h_texcoord = vec2(
		mirror_repeat(v_texcoord.x, repeat.x, mirror_x)*texture_scale.x,
		mirror_repeat(v_texcoord.y, repeat.y, mirror_y)*texture_scale.y
	);

        vec4 output_color = texture2D(tex, h_texcoord);
	if (apply_sobel==1){
		vec2 grad = getTextureIntensityGradient(tex, h_texcoord, u_resolution);
		output_color = output_color*length(grad);
	}	
	if(is_crosshatch==1) output_color.rgb = crosshatch(output_color.rgb);
	if(halftone_resolution>0) output_color.rgb = halftone(output_color.rgb, h_texcoord*halftone_resolution);
	if(is_inverted==1) output_color.rgb = 1 - output_color.rgb;
	
	gl_FragColor = vec4(output_color.rgb, output_color.a*transparency)*brightness;
}
