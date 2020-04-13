from graphviz import Digraph

import renderer.pipeline.nodes as N
from renderer.pipeline.graph import NodeOutput
from renderer.pipeline.graph import Preset
from renderer.pipeline.graph import RenderGraph
from renderer.pipeline.graph import TogglePreset


def visualize_graph(graph, name="graphviz_graph"):
    important_nodes = ["draw", "geo", "shader", "bpm_clock", "biggan"]
    digraph = Digraph(name)
    for name, _ in graph._nodes.items():
        if name in important_nodes:
            shape = "doublecircle"
        else:
            shape = "circle"
        digraph.attr("node", shape=shape)
        digraph.node(name, name)
    reverse_node_dict = {v: k for k, v in graph._nodes.items()}
    for name, node in graph._nodes.items():
        for key, value in node.inputs().items():
            in_edge = value.get_link()
            if isinstance(in_edge, NodeOutput):
                start = reverse_node_dict[in_edge.attached_node]
                digraph.edge(start, name, label=key)
    digraph.render()


def simple_gan_setup(model_type="biggan-deep-128"):
    g = RenderGraph()
    g.add_node(N.Window(), name="window")
    g.add_node(
        N.BpmClock(bpm=120), name="bpm_clock",
    )
    g.add_node(N.ClassPool(), name="class_pool")
    g.add_node(
        N.ClassSampler(class_pool=g["class_pool"].output_class_pool),
        name="class_sampler",
    )
    g.add_node(
        N.SampleManager(
            time=g["bpm_clock"].local_time, sampler=g["class_sampler"].sampler,
        ),
        name="class",
    )
    g.add_node(
        N.Biggan(class_vector=g["class"].vector0, model_type=model_type,),
        name="biggan",
    )
    g.add_node(
        N.Shader(), name="shader",
    )
    g.add_node(
        N.GeometryController(
            shader=g["shader"].shader,
            texture=g["biggan"].image,
            window_size=g["window"].size,
        ),
        name="geo",
    )

    g.add_node(
        N.DrawNode(
            faces=g["geo"].faces, shader=g["shader"].shader, window=g["window"].window
        ),
        name="draw",
    )
    g.set_draw_node("draw")
    return g


def complex_gan_setup(model_type="biggan-deep-128"):
    g = RenderGraph()
    add_default_nodes(g)
    g.add_node(
        N.ComplexFunction(x=g["bpm_clock"].normalized_time, y=1.0, function="x"),
        name="noise_timing",
    )
    g.add_node(N.NoiseSampler(sample_options="SimpleRotation"), name="noise_sampler")
    g.add_node(
        N.ComplexFunction(
            x=g["bpm_clock"].normalized_time,
            single_beat_length=16.0,
            function="x",
            presets=["0.5-cos(x*3*pi)", "0"],
        ),
        name="class_timing",
    )
    g.add_node(N.ClassPool(), name="class_pool")
    g.add_node(
        N.ClassSampler(class_pool=g["class_pool"].output_class_pool),
        name="class_sampler",
    )
    g.add_node(
        N.SampleManager(
            time=g["noise_timing"].linear_result, sampler=g["noise_sampler"].sampler
        ),
        name="noise",
    )
    g.add_node(
        N.SampleManager(
            time=g["class_timing"].linear_result, sampler=g["class_sampler"].sampler,
        ),
        name="class",
    )

    g.add_node(
        N.Interpolate(
            interpolation_value=g["noise_timing"].result,
            input_vector0=g["noise"].vector0,
            input_vector1=g["noise"].vector1,
        ),
        name="noise_interpolate",
    )
    g.add_node(
        N.Interpolate(
            interpolation_value=g["class_timing"].result,
            input_vector0=g["class"].vector0,
            input_vector1=g["class"].vector1,
        ),
        name="class_interpolate",
    )
    g.add_node(
        N.Biggan(
            noise_vector=g["noise_interpolate"].result,
            class_vector=g["class_interpolate"].result,
            model_type=model_type,
        ),
        name="biggan",
    )
    g.add_node(
        N.ComplexFunction(
            x=g["bpm_clock"].normalized_time,
            y=0.25,
            function="1",
            presets=[
                "1",
                "x",
                "y",
                "x**y*(a)+(1-a)",
                "x**y*(a+b*sin(c*x*pi))+(1-(a+b*sin(c*x*pi)))",
            ],
            bounds=(0, 5),
        ),
        name="texture_timing",
    )
    g.add_node(
        N.ComplexFunction(
            x=g["bpm_clock"].normalized_time, y=0, function="100", bounds=(0, 10000)
        ),
        name="sobel_timing",
    )
    g.add_node(
        N.ComplexFunction(
            x=g["bpm_clock"].normalized_time, y=0, function="y", bounds=(0, 10000)
        ),
        name="halftone_timing",
    )
    g.add_node(
        N.ComplexFunction(x=g["bpm_clock"].normalized_time, y=1, function="y"),
        name="brightness_timing",
    )
    g.add_node(
        N.Shader(
            texture_scaling_x=g["texture_timing"].result,
            texture_scaling_y=g["texture_timing"].result,
            sobel_resolution=g["sobel_timing"].result,
            halftone_resolution=g["halftone_timing"].result,
            brightness=g["brightness_timing"].result,
        ),
        name="shader",
    )
    g.add_node(N.ModelMatrix(), name="model_matrix")
    g.add_node(
        N.GeometryController(
            shader=g["shader"].shader,
            texture=g["biggan"].image,
            mesh="quad",
            model_matrix=g["model_matrix"].model_matrix,
            window_size=g["window"].size,
        ),
        name="geo",
    )
    g.add_node(
        N.RecordingDrawNode(
            faces=g["geo"].faces, shader=g["shader"].shader, window=g["window"].window
        ),
        name="draw",
    )
    g.set_draw_node("draw")

    g.add_preset("default", Preset({k: v.value for k, v in g.default_inputs().items()}))
    g.add_preset(
        "Sobel_mode",
        TogglePreset(
            on={"shader.apply_sobel": True}, off={"shader.apply_sobel": False}
        ),
    )

    g.add_preset(
        "Mirrored",
        TogglePreset(
            on={"shader.is_mirrored": (True, True)},
            off={"shader.is_mirrored": (False, False)},
        ),
    )
    g.add_preset(
        "cone",
        TogglePreset(
            on={
                "geo.mesh": "empty_cone",
                "shader.repeat_times": (1.5, 1.5),
                "shader.is_mirrored": (True, True),
            },
            off={
                "geo.mesh": "quad",
                "shader.repeat_times": (0.0, 0.0),
                "shader.is_mirrored": (False, False),
            },
        ),
    )

    g.add_preset(
        "Fractal_Vegetables", Preset({"class_pool.class_pool": [937, 938, 944]})
    )
    g.add_preset(
        "Round_objects", Preset({"class_pool.class_pool": [712, 409, 835, 971]})
    )
    g.add_preset(
        "Apes_n_Monkeys",
        Preset(
            {
                "class_pool.class_pool": [
                    369,
                    367,
                    378,
                    380,
                    376,
                    382,
                    379,
                    372,
                    377,
                    374,
                    371,
                    373,
                    381,
                    375,
                    370,
                    365,
                    366,
                    368,
                ]
            }
        ),
    )

    g.add_preset(
        "Slow_changes",
        Preset(
            {
                "noise_timing.single_beat_length": 1.0,
                "noise_timing.function": "x**y",
                "noise_timing.y": 1.0,
                "class_timing.single_beat_length": 32.0,
                "class_timing.function": "x**y",
                "class_timing.y": 1.0,
                "noise_sampler.sample_options": "ComplexRotation",
                "noise_sampler.rotation_speed": 0.03,
                "noise_sampler.truncation": 1.0,
            }
        ),
    )
    g.add_preset(
        "Fast_Snappy_Noise",
        Preset(
            {
                "noise_sampler.sample_options": "PingPong",
                "noise_sampler.rotation_speed": 0.1,
                "noise_sampler.truncation": 0.5,
                "noise_timing.single_beat_length": 1.0,
                "noise_timing.function": "x**y",
                "noise_timing.y": 8.0,
            }
        ),
    )

    g.add_preset(
        "funky_spectrals",
        Preset(
            {
                "texture_timing.single_beat_length": 1.0,
                "texture_timing.y": 0.25,
                "texture_timing.z": 1.0,
                "texture_timing.abc": [1.0, 1.0, 1.0],
                "texture_timing.function": "1",
                "sobel_timing.single_beat_length": 1.0,
                "sobel_timing.y": 1.0,
                "sobel_timing.z": 1.0,
                "sobel_timing.abc": [1.0, 1.0, 1.0],
                "sobel_timing.function": "2*y+x*100*z",
                "shader.apply_sobel": True,
                "shader.is_mirrored": [True, True],
                "shader.repeat_times": [1.0, 1.0],
            }
        ),
    )

    g.add_preset(
        "lights_blinking",
        Preset({"brightness_timing.function": "y-(x*y)", "brightness_timing.y": 0.3}),
    )
    g.add_preset(
        "lights_on",
        Preset({"brightness_timing.function": "y", "brightness_timing.y": 0.3}),
    )
    g.add_preset(
        "lights_out",
        Preset({"brightness_timing.function": "y", "brightness_timing.y": 0.0}),
    )
    return g


def no_gan_setup():
    g = RenderGraph()
    add_default_nodes(g)

    g.add_node(
        N.BinaryFunction(x=g["bpm_clock"].local_time, y=4, function="x**y"),
        name="power",
    )
    g.add_node(N.PaddedColorToImage(), name="img0")
    g.add_node(N.PaddedColorToImage(color=(0, 0, 0)), name="img1")
    g.add_node(
        N.Interpolate(
            interpolation_value=g["bpm_clock"].delta_time,
            input_vector0=g["img0"].image,
            input_vector1=g["img1"].image,
        ),
        name="img",
    )
    g.add_node(
        N.BinaryFunction(x=g["bpm_clock"].counter, y=8, function="x % y"),
        "mirror_clock",
    )
    g.add_node(
        N.ComplexFunction(
            x=g["bpm_clock"].normalized_time,
            y=8,
            function="x*y",
            bounds=(0, 100),
            single_beat_length=4,
        ),
        name="repeat",
    )
    g.add_node(
        N.ScalarToVector(scalar=g["repeat"].result, vector_size=2), name="repeat2"
    )
    g.add_node(
        N.Shader(is_mirrored=(True, True), repeat_times=g["repeat2"].vector),
        name="shader",
    )
    g.add_node(
        N.ComplexFunction(
            x=g["bpm_clock"].normalized_time,
            function="x*180",
            single_beat_length=16,
            bounds=(0, 180),
        ),
        name="rotation",
    )
    g.add_node(
        N.ScalarToVector(scalar=g["rotation"].result, vector_size=3), name="rotation3"
    )

    g.add_node(N.ModelMatrix(rotation=g["rotation3"].vector), name="model_matrix")
    g.add_node(
        N.GeometryController(
            shader=g["shader"].shader,
            texture=g["img"].result,
            mesh="octahedron",
            window_size=g["window"].size,
            view_distance=-4,
            model_matrix=g["model_matrix"].model_matrix,
        ),
        name="geo",
    )
    g.add_node(
        N.DrawNode(
            faces=g["geo"].faces, shader=g["shader"].shader, window=g["window"].window
        ),
        name="draw",
    )
    g.set_draw_node("draw")
    return g


def add_default_nodes(g: RenderGraph, bpm=120):
    g.add_node(N.Window(), name="window")
    g.add_node(
        N.BpmClock(bpm=bpm), name="bpm_clock",
    )
    return g


def debug_graph():
    g = RenderGraph()
    g["window"] = N.Window()
    g["bpm_clock"] = N.BpmClock(bpm=30)
    g["white"] = N.ColorToImage(color=(1, 1, 1))
    g["black"] = N.ColorToImage(color=(0, 0, 0))
    g["output_color"] = N.Interpolate(
        interpolation_value=g["bpm_clock"].local_time,
        input_vector0=g["white"].image,
        input_vector1=g["black"].image,
    )
    g["shader"] = N.Shader()
    g["geo"] = N.GeometryController(
        shader=g["shader"].shader,
        texture=g["output_color"].result,
        mesh="quad",
        window_size=g["window"].size,
    )
    g["draw"] = N.DrawNode(
        faces=g["geo"].faces, shader=g["shader"].shader, window=g["window"].window
    )
    g.set_draw_node("draw")
    return g
