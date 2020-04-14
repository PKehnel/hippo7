uniform float scale;
attribute vec3 position;
uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;
attribute vec2 texcoord;
varying vec2 v_texcoord;
void main()
{
    v_texcoord = texcoord;
    gl_Position = projection * view * model * vec4(scale*position, 1.0);
}