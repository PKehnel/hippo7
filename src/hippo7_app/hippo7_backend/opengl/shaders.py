from pathlib import Path


def get_shader(file):
    cwd = Path(__file__).parent.absolute()
    file = cwd / "shaders" / file
    with file.open() as shader:
        return shader.read()


def get_fragment_shader():
    fragment = """
    uniform sampler2D tex;
    uniform int mirror_times;
    varying vec2 v_texcoord;
    void main()
    {
        vec2 h_texcoord = v_texcoord;
        for(int i=0;i < mirror_times;++i){
            h_texcoord = max(1 - abs(h_texcoord*2-1), 0.);
        }
        gl_FragColor = texture2D(tex, h_texcoord);
    } """
    return fragment


def get_vertex_shader():
    vertex = """
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
    } """
    return vertex
