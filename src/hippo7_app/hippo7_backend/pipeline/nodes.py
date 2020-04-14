import math
from datetime import datetime
from math import cos
from math import sin
from random import choice

import numexpr
import numpy as np
import torch
import torch.nn.functional as f
from glumpy import gl
from glumpy import glm
from glumpy import gloo

from hippo7_app.hippo7_backend import get_asset_folder
from hippo7_app.hippo7_backend.opengl.geometry import correctly_rotated_model_matrix, to_tex_format, load_file
from hippo7_app.hippo7_backend.opengl.render import WindowManager
from hippo7_app.hippo7_backend.opengl.shaders import get_shader
from hippo7_app.hippo7_backend.pipeline import song_dict, mesh_dict
from hippo7_app.hippo7_backend.pipeline.graph import Node
from hippo7_app.hippo7_backend.pipeline.graph import NodeInput
from hippo7_app.hippo7_backend.pipeline.graph import NodeOutput
from hippo7_app.hippo7_backend.pipeline.graph import T
from hippo7_app.hippo7_backend.pipeline.nodes_util import BeatTimeMessage, create_noise_vector, create_class_vector, \
    generate_model_file, toNumpy

from hippo7_app.hippo7_backend.pipeline.nodes_util import time_generator


class DrawNode(Node):
    """
    The draw node is the standard final node in a graph to display something. All three
    inputs must be connected.

    Inputs:
        faces:
            A list of the faces that should be displayed. This needs to come from a
            GeometryController Node as that node transmits the actual mesh data to the shader.
        shader:
            A ShaderProgram that runs the rendering pipeline.
        window:
            The window that the frame should be rendered on.

    Outputs:
        The DrawNode has no output, since it renders directly to the window.
    """

    faces = NodeInput([0], ntype=T.List(T.UInt()))
    shader = NodeInput(ntype=T.ShaderProgram())
    window = NodeInput(ntype=T.Window())

    def update(self):

        # draw to screen
        self.window.clear()

        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        self.shader.draw(gl.GL_TRIANGLES, self.faces)


class RecordingDrawNode(Node):
    """
        The recording draw node is the a final node in a graph to both display and record
        something. All three inputs faces, shader and window must be connected.

        Inputs:
            faces:
                A list of the faces that should be displayed. This needs to come from a
                GeometryController Node as that node transmits the actual mesh data to the shader.
            shader:
                A ShaderProgram that runs the rendering pipeline.
            window:
                The window that the frame should be rendered on.

        Outputs:
            recording:
                A bool if a video should be recorded or not.
            song_name:
                The song that should be the background music for the recorded video.
        """

    faces = NodeInput([0], ntype=T.List(T.UInt()))
    shader = NodeInput(ntype=T.ShaderProgram())
    window = NodeInput(ntype=T.Window())
    recording = NodeInput(False, ntype=T.Bool())
    song_name = NodeInput("", ntype=T.FixedSelector(T.String(), song_dict.keys(),),)

    def __init__(
        self, video_folder=get_asset_folder() / "videos", **kwargs,
    ):
        super(RecordingDrawNode, self).__init__(**kwargs)
        self.currently_recording = False
        self.video_folder = video_folder
        self.fps = 15
        self.recording_time = 0

    def write_current_frame(self):
        gl.glReadPixels(
            0,
            0,
            self.window.width,
            self.window.height,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            self.frame_buffer,
        )
        self.writer.write_frame(np.flipud(self.frame_buffer))

    def update(self):
        self.window.clear()
        gl.glDisable(gl.GL_BLEND)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_POLYGON_OFFSET_FILL)
        self.shader.draw(gl.GL_TRIANGLES, self.faces)
        self.check_recording()

    def check_recording(self):
        if self.recording:
            # start a new stream
            if not self.currently_recording:
                self.recording_time = 0
                self.currently_recording = True
                self.writer, self.frame_buffer = self.declare_writer()
            if self.recording_time % self.fps == 0:
                print(f"recording time: {int(self.recording_time / self.fps)}s")
            self.recording_time += 1
            self.write_current_frame()
        else:
            if self.currently_recording:
                self.currently_recording = False
                self.writer.close()

    def declare_writer(self):
        """
        Create a ffmpeg writer and define the output file name.
        For this to work a change to the FFMPEG_VideoWriter backend has to be done:

        - search and remove -an
        - optional so the song works: add '-shortest' to the audio option
        - https://github.com/Zulko/moviepy/pull/968 if merged maybe this is not needed anymore

        Returns:
            Ffmpeg writer and the frame buffer.

        """
        shape = 512
        try:
            audio_file = song_dict[self.song_name]
        except KeyError:
            audio_file = None
        from glumpy.ext.ffmpeg_writer import FFMPEG_VideoWriter  # NOQA

        self.output_file_name = (
            str(self.video_folder)
            + f"/{self.song_name}_"
            + datetime.now().strftime("%d:%m_%H:%M")
            + ".mp4"
        )

        ffpmeg_vw = FFMPEG_VideoWriter(
            filename=self.output_file_name,
            size=(shape, shape),
            fps=self.fps,
            preset="veryfast",
            codec="libx264",  # h264
            audiofile=audio_file,
        )
        frame_buffer = np.zeros((shape, shape, 3), dtype=np.uint8)
        return ffpmeg_vw, frame_buffer


class GeometryController(Node):
    """
    The geometry controller node is one of the key nodes to render something. It manages
    mesh creation, texturing and mesh positioning.

    Inputs:
        shader:
            A shader program that the mesh data will be transmitted to.
        texture:
            A texture that will be displayed on the mesh.
        mesh:
            The mesh that will be displayed. Currently all available meshes are in the
            assets/meshes/ folder and are selectable in the frontend. You can access the path
            list through mesh_dict.
        view_distance:
            The position of the camera. It is currently only movable in the z
            direction.
        window_size:
            The size of the window, should be bound the output of a Window node.
        model_matrix:
            All parameters for the positioning, scaling and rotation of the mesh.
            Currently there are no extra measures against gimbal locking.

    Outputs:
        faces:
            This output is a list of face ids that should be displayed. It should be
            connected to a faces input on one of the recording nodes.
    """

    shader = NodeInput(ntype=T.ShaderProgram())
    texture = NodeInput(torch.zeros(1, 3, 1, 1), ntype=T.Image())
    mesh = NodeInput("quad", ntype=T.FixedSelector(T.String(), mesh_dict.keys(),),)
    view_distance = NodeInput(-2.5, ntype=T.Float(-30, 5))
    window_size = NodeInput(ntype=T.Tuple(T.UInt(), T.UInt()))
    model_matrix = NodeInput(correctly_rotated_model_matrix(), ntype=T.Tensor((4, 4)))

    faces = NodeOutput(ntype=T.List(T.UInt()))

    def __init__(self, **kwargs):
        super(GeometryController, self).__init__(**kwargs)
        self.internal_mesh = ""

    def update(self):
        if self.mesh != self.internal_mesh:
            self.load_mesh(mesh_dict[self.mesh])
            self.internal_mesh = self.mesh

        # The model is rotated because the texturing is the wrong way around
        self.shader["model"] = self.model_matrix
        self.shader["view"] = glm.translation(0, 0, self.view_distance)
        # Scale is increased to fill the whole screen
        self.shader["scale"] = 1.0
        self.shader["tex"] = to_tex_format(self.texture)
        width, height = self.window_size
        self.shader["projection"] = glm.perspective(
            # 45.0, 1.0, 2.0, 100.0
            45.0,
            width / float(height),
            2.0,
            100.0,
        )

    def load_mesh(self, file):
        vertices, faces = load_file(file)
        self.faces = faces
        self.shader.bind(vertices)
        self.has_update = False


class PaddedColorToImage(Node):
    """
    Takes a color input and creates a 4x4 texture from it, where one corner is assigned
    a color, while the rest is black.

    Inputs:
        color:
            A color tuple input.

    Outputs:
        image:
            A valid picture of the size 4x4, which can be used as a texture or for
            something else.
    """

    color = NodeInput((1, 1, 1), ntype=T.RGB())

    image = NodeOutput(ntype=T.Image())

    def update(self):
        image = (
            (torch.tensor(self.color) * 2 - 1)
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(2)
            .float()
            .to(self.device)
        )
        self.image = f.pad(image, (1, 0, 1, 0), value=-1)


class ColorToImage(Node):
    """
    Takes a color input and creates a valid texture from it.

    Inputs:
        color:
            A color tuple input.

    Outputs:
        image:
            A valid picture of the size 1x1, which can be used as a texture or for
            something else.
    """

    color = NodeInput((1, 1, 1), ntype=T.RGB())

    image = NodeOutput(ntype=T.Image())

    def update(self):
        image = (
            (torch.tensor(self.color) * 2 - 1)
            .unsqueeze(0)
            .unsqueeze(2)
            .unsqueeze(2)
            .float()
            .to(self.device)
        )
        self.image = image


class BpmClock(Node):
    """
    The BpmClock is one of the core nodes for a standard Rendergraph. It gives out
    multiple signals synchronized to a bpm, typically the bpm of the song that is
    currently playing.

    Inputs:
        bpm:
            A beats per minute value.

    Outputs:
        beat_time:
            An object that contains all information outputted by the node in one
            place.
        counter:
            A counter of the beats, so 10 will mean that 10 beats have passed in total.
        isBeat:
            If the current frame is on a beat or not.
        time:
            Total time that has passed since startup.
        normalized_time:
            The time normalized by the bpm, so that each unit is equal to the
            time passing between two beats. So 12.20 will be the value after 12 beats passing and
            20 % along the way to the next beat. Is equal to counter + local_time.
        delta_time:
            The time passed between two frames.
        local_time:
            The percent of time passed between the last and next beat. Basically, it
            is a value between 0 and 1 that describes how far we are distanced to both on a time
            basis.
    """

    bpm = NodeInput(128.0, ntype=T.Float(lower_bound=0.00001, upper_bound=300))

    beat_time = NodeOutput(BeatTimeMessage(), ntype=T.BeatTime())
    counter = NodeOutput(0)
    isBeat = NodeOutput(False)
    time = NodeOutput(0.0)
    normalized_time = NodeOutput(0.0)
    delta_time = NodeOutput(0.0)
    local_time = NodeOutput(0.0)

    def __init__(self, time_generator=time_generator(), **kwargs):
        super().__init__(**kwargs)
        self.time_gen = time_generator
        self.internal_time = 0
        self.internal_counter = 0
        self.total_time = 0
        self.init_time = next(self.time_gen)
        self.last_time = self.init_time
        self.last_local_time = 0

    def update(self):
        time = next(self.time_gen) - self.init_time
        counter, local_time = divmod(time, 60 / self.bpm)
        local_time *= self.bpm / 60
        isBeat = self.last_local_time > local_time
        self.last_local_time = local_time
        delta_time = time - self.last_time
        self.last_time = time
        bt = BeatTimeMessage(
            counter=counter,
            isBeat=isBeat,
            time=time,
            delta_time=delta_time,
            local_time=local_time,
            bpm=self.bpm,
            speed=1,
        )
        self.counter = counter
        self.isBeat = isBeat
        self.time = time
        self.delta_time = delta_time
        self.local_time = local_time
        self.beat_time = bt
        self.normalized_time = local_time + counter


class Interpolate(Node):
    """
    Interpolates between two vectors. The two vectors should have the same dimensions or
    support pytorch broadcasting.

    Inputs:
        interpolation_value:
            Where betweeen the two vectors the output should be set to.
            Values between 0 and 1 make the most sense, but others are also valid and represent
            an extrapolation.
        input_vector0:
            The first vector.
        input_vector1:
            The second vector.

    Outputs:
        result:
            The interpolation result.

    """

    interpolation_value = NodeInput(ntype=T.Float())
    input_vector0 = NodeInput(torch.zeros(1), ntype=T.Tensor())
    input_vector1 = NodeInput(torch.ones(1), ntype=T.Tensor())

    result = NodeOutput(ntype=T.Tensor())

    def update(self):
        self.result = torch.lerp(
            self.input_vector0, self.input_vector1, float(self.interpolation_value)
        )


class Biggan(Node):
    """
    The Biggan Node generates pictures based on a noise_vector and a class_vector(These
    can be generated by the NoiseSampler/ClassSampler Nodes).

    Inputs:
        noise_vector:
            A noise vector in the shape of (1, 128).
        class_vector:
            A class vector in the shape of (1, 1).

    Outputs:
        image:
            A generated output image.
    """

    noise_vector = NodeInput(
        create_noise_vector(batch_size=1, truncation=1, device="cpu"), ntype=T.Tensor()
    )
    class_vector = NodeInput(
        create_class_vector(643, batch_size=1, device="cpu"), ntype=T.Tensor()
    )

    image = NodeOutput(ntype=T.Tensor())

    def __init__(self, model_type: str = "biggan-deep-128", **kwargs):
        super(Biggan, self).__init__(**kwargs)
        self.model = self.load_model(model_type, device=self.device)
        self.truncation = torch.ones(1, device=self.device)
        self.model_type = model_type

    def update(self):
        with torch.no_grad():
            self.image = self.model(
                self.noise_vector, self.class_vector, self.truncation
            )

    def to_device(self, device: str):
        if device != self.device:
            self.model = self.load_model(
                self.model_type, device="cuda" if device.startswith("cuda") else device
            )

        super().to_device(device)
        self.truncation = self.truncation.to(device)

    def load_model(self, model_type, device):
        file_name = str(get_asset_folder() / "networks" / (model_type + "-" + device))
        try:
            return torch.jit.load(file_name)
        except (ValueError, RuntimeError):
            print(
                "Could not find",
                file_name,
                "\n Trying to create file(requires network connection)",
            )
            generate_model_file(model_type, device)
            return torch.jit.load(file_name)


class Shader(Node):
    """
    The shader node manages the Vertex and Fragment Shader of the pipeline. It has
    numerous inputs that manage different effects. It does not manage the geometry or
    texture parts of the shader inputs, that is the mesh, its position or texture.

    Inputs:
        apply_sobel:
            If True, applies edge detection through the sobel operator.
        sobel_resolution:
            The raster resolution for the sobel operator. The higher, the
            finer the edge results.
        is_mirrored:
            If True, mirrors the texture along the (u,v) axes.
        repeat_times:
            How often the texture should be repeated in the (u,v) coordinate space.
        texture_scaling_x:
            Multiplies the texture coordinates in the u dimension.
        texture_scaling_y:
            Multiplies the texture coordinates in the v dimension.
        is_crosshatched:
            Applies the crosshatching effect on the texture.
        is_inverted:
            Inverts the output colors.
        halftone_resolution:
            If >0, applies the halftone effect. The greater the value, the
            finer the resolution.
        brightness:
            Multiplies the pixel output by this value to control pixel brightness.

    Outputs:
        shader:
            The shader program as an output, needed for both the GeometryController and
            the DrawNode.
    """

    apply_sobel = NodeInput(False, ntype=T.Bool())
    sobel_resolution = NodeInput(512.0, ntype=T.Float(0.1, 1000))

    is_mirrored = NodeInput((False, False), ntype=T.Tuple(T.Bool(), T.Bool()))
    repeat_times = NodeInput(
        [1.0, 1.0], ntype=T.Vector2(0, 10.0), pre_bind_hook=toNumpy
    )
    texture_scaling_x = NodeInput(1.0)
    texture_scaling_y = NodeInput(1.0)

    is_crosshatched = NodeInput(False, ntype=T.Bool())
    is_inverted = NodeInput(False, ntype=T.Bool())
    halftone_resolution = NodeInput(0.0, ntype=T.Float(0, 10))

    brightness = NodeInput(1.0, ntype=T.Float(0.0, 1.0))
    # transparency = NodeInput(1.0, ntype=T.Float(0.0, 1.0))

    shader = NodeOutput(ntype=T.ShaderProgram())

    def __init__(self, **kwargs):
        super(Shader, self).__init__(**kwargs)
        vertex = get_shader("vertex.glsl")
        fragment = get_shader("fragment.glsl")
        self.internal_shader = gloo.Program(vertex, fragment)
        self.shader = self.internal_shader

    def update(self):
        self.internal_shader["apply_sobel"] = self.apply_sobel
        self.internal_shader["u_resolution"] = self.sobel_resolution
        self.internal_shader["mirror_x"] = self.is_mirrored[0]
        self.internal_shader["mirror_y"] = self.is_mirrored[1]
        self.internal_shader["repeat"] = self.repeat_times
        self.internal_shader["texture_scale"] = np.array(
            [self.texture_scaling_x, self.texture_scaling_y], dtype=float
        )

        self.internal_shader["is_crosshatch"] = self.is_crosshatched
        self.internal_shader["is_inverted"] = self.is_inverted
        self.internal_shader["halftone_resolution"] = self.halftone_resolution

        self.internal_shader["brightness"] = self.brightness
        self.internal_shader["transparency"] = 1  # self.transparency

        self.shader = self.internal_shader


class Window(Node):
    """
    This node exists for resizing and rendering purposes. It provides access to both the
    window size and the window.

    Outputs:
        size:
            The size of the window as a (width, height) tuple.
        window:
            The window object.
    """

    size = NodeOutput(ntype=T.Tuple(T.UInt(), T.UInt()))
    window = NodeOutput(ntype=T.Window())

    def __init__(self, **kwargs):
        super(Window, self).__init__(**kwargs)
        self.internal_window = WindowManager.get_window()
        self.window = self.internal_window
        self.size = self.internal_window.width, self.internal_window.height

    def update(self):
        self.size = WindowManager.get_window_size()


class BinaryFunction(Node):
    """
    Calculates the result of two input values x y and an arbitrary function string. It
    is preferable to use the ComplexFunction Node as it provides additional values as
    inputs and has better error checking.

    Inputs:
        x:
            A float input.
        y:
            A float input.
        function:
            A function that should be computed at every timestep. Should be provided as
            a string.

    Outputs:
        result:
            Result of the computation.

    """

    x = NodeInput(1.0, ntype=T.Float(-10, 10))
    y = NodeInput(1.0, ntype=T.Float(-10, 10))
    function = NodeInput("x+y", ntype=T.MathFunction())

    result = NodeOutput(ntype=T.Float())

    def update(self):
        self.result = numexpr.evaluate(
            self.function, local_dict={"x": self.x, "y": self.y}
        )


class ModifiedBpm(Node):
    """
    The ModifiedBpm node is used to modify bpm speed and then provide the same outputs.

    Inputs:
        in_beat_time:
            A BeatTimeMessage object to modifiy.
        time_modifier:
            The speed value by which the time should be changed.

    Outputs:
        out_beat_time:
            An object that contains all information outputted by the node in one
            place.
        counter:
            A counter of the beats, so 10 will mean that 10 beats have passed in total.
        isBeat:
            If the current frame is on a beat or not.
        time:
            Total time that has passed since startup.
        normalized_time:
            The time normalized by the bpm, so that each unit is equal to the
            time passing between two beats. So 12.20 will be the value after 12 beats
            passing and 20 % along the way to the next beat. Is equal to counter +
            local_time.
        delta_time:
            The time passed between two frames.
        local_time:
            The percent of time passed between the last and next beat. Basically, it
            is a value between 0 and 1 that describes how far we are distanced to both
            on a time basis.
    """

    in_beat_time = NodeInput(ntype=T.BeatTime())
    time_modifier = NodeInput(1.0, ntype=T.Float(0, 10))

    out_beat_time = NodeOutput(ntype=T.BeatTime())
    counter = NodeOutput(0)
    isBeat = NodeOutput(False)
    time = NodeOutput(0)
    delta_time = NodeOutput(0)
    local_time = NodeOutput(0)
    normalized_time = NodeOutput(0)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_local_time = 0

    def update(self):
        bpm = self.in_beat_time.bpm
        time = self.in_beat_time.time * self.time_modifier
        counter, local_time = divmod(time, 60 / bpm)
        local_time *= bpm / 60
        isBeat = self.last_local_time > local_time
        self.last_local_time = local_time
        delta_time = self.in_beat_time.delta_time * self.time_modifier

        bt = BeatTimeMessage(
            counter=counter,
            isBeat=isBeat,
            time=time,
            delta_time=delta_time,
            local_time=local_time,
            bpm=bpm,
            speed=self.time_modifier,
        )
        self.counter = counter
        self.isBeat = isBeat
        self.time = time
        self.delta_time = delta_time
        self.local_time = local_time
        self.out_beat_time = bt
        self.normalized_time = counter + local_time


class ClassPool(Node):
    """
    A class pool node, which has additional frontend logic to make the class ids are
    properly selectable.

    Inputs:
        class_pool:
            The Classes that should be used for possible interpolation targets.

    Outputs:
        output_class_pool:
            The value to access the class pool.
    """

    class_pool = NodeInput(list(range(0, 999)), ntype=T.List(T.UInt()))
    output_class_pool = NodeOutput(ntype=T.List(T.UInt()))

    def __init__(self, **kwargs):
        super(ClassPool, self).__init__(**kwargs)
        self.output_class_pool = self.class_pool

    def update(self):
        self.output_class_pool = self.class_pool


class ClassSampler(Node):
    """
    Samples classes as defined in the sampling scheme.

    Inputs:
        class_pool:
            Pool of classes we sample from.
        sample_options:
            The sampling scheme that will be used. Currently avaiable schemes are: "star": Samples ray_count
            nodes in consecutive order before resampling the center. The center is normally the class pool at index 0,
            but can be changed with resample_center_every_n:

            "continuous":
                Samples the class_pool in round robin fashion.
            "ping_pong":
                Switches between the 0th and 1st entry of the class pool.
            "random":
                Samples a random class from the pool. Guarantees that no class while be
                sampled 2 times in a row.
        resample_center_every_n:
            The bool decides if it the center in star mode should be resampled every n values,
            the int is the n.
        ray_count:
            How many rays should be sampled before jumping back to the center.

    Outputs:
        sampler:
            The sample function that can be used to sample a value.
    """

    class_pool = NodeInput(list(range(0, 999)), ntype=T.List(T.UInt()))
    sample_options = NodeInput("Random", ntype=T.FixedSelector(T.String(), ["Star","Continuous","Ping Pong","Random"]),)
    resample_center_every_n = NodeInput(
        (False, 8), ntype=T.Tuple(T.Bool(), T.UInt(lower_bound=1, upper_bound=15))
    )
    ray_count = NodeInput(1, ntype=T.UInt(lower_bound=1, upper_bound=15))
    sampler = NodeOutput(ntype=T.Sampler())

    def __init__(self, **kwargs):
        super(ClassSampler, self).__init__(**kwargs)
        self.sampler = self.sample
        self.sample_schema = {
            "Star": ClassSampler.Star,
            "Continuous": ClassSampler.Continuous,
            "Ping Pong": ClassSampler.PingPong,
            "Random": ClassSampler.Random,
        }
        self._in_edges["sample_options"].ntype.known_values = self.sample_schema.keys()
        self.step_counter = 0
        self.inner_option = self.sample_options
        self.sample_mode = self.switch_mode()

    def update(self):
        if self.inner_option != self.sample_options:
            self.inner_option = self.sample_options
            self.sample_mode = self.switch_mode()

    def sample(self):
        self.step_counter += 1
        if len(self.class_pool) == 1:
            self.class_label = self.class_pool[0]
        else:
            self.class_label = self.sample_mode.generate(
                self.class_pool,
                self.step_counter,
                self.resample_center_every_n,
                self.ray_count,
            )
        return create_class_vector(self.class_label, batch_size=1, device=self.device)

    def switch_mode(self):
        return self.sample_schema[self.sample_options](self.class_pool)

    class Random:
        def __init__(self, class_pool):
            self.class_label = -1

        def generate(self, class_pool, *args, **kwargs):
            class_label = self.class_label
            while class_label == self.class_label:
                self.class_label = choice(class_pool)
            return self.class_label

    class Star:
        def __init__(self, class_pool):
            self.star_center = self.sample_class_vector = choice(class_pool)

        def generate(
            self, class_pool, step_counter, resample_center_every_n, ray_count
        ):
            resample, every_n = resample_center_every_n
            if resample:
                if step_counter % (every_n * (ray_count + 1)) == 0:
                    self.star_center = choice(class_pool)
            else:
                self.star_center = class_pool[0]
            if step_counter % (ray_count + 1) == 0:
                sample_class_vector = self.star_center
            else:
                sample_class_vector = class_pool[
                    1 + int(step_counter / 2) % (len(class_pool) - 1)
                ]
            return sample_class_vector

    class PingPong:
        def __init__(self, class_pool):
            pass

        def generate(
            self, class_pool, step_counter, resample_center_every_n, ray_count
        ):
            return class_pool[step_counter % 2]

    class Continuous:
        def __init__(self, class_pool):
            pass

        def generate(
            self, class_pool, step_counter, resample_center_every_n, ray_count
        ):
            return class_pool[step_counter % len(class_pool)]


class NoiseSampler(Node):
    """
    Samples noise as defined by the sampling scheme.

    Inputs:
        sample_options:
            The mode how the noise will be sampled. All samples are drawn from a
            normal distribution, if not specified. Possible modes are:

            "PingPong":
                Samples a vector and then inverts it on every consecutive sampling.
            "SimpleRotation":
                Samples the center(so all zeroes), except the 0th and first
                dimension are sampled along a circle, the amount of rotation per sample is
                given by the rotation_speed value.
            "ComplexRotation":
                Samples in a very chaotic rotation roughly around the center.
            "Random":
                Samples in a random manner.
            "RandomStar":
                Samples randomly, but every second sample is all zeroes.
        truncation:
            Multiplies the vector by a value, so the higher the truncation, the
            higher the likelihood that the sample is nonsensical, the lower the truncation,
            the smaller the sample variance.
        rotation_speed:
            Speed of rotation around the center for the two rotation sampling
            modes. The higher, the faster the samples change.

    Outputs:
        sampler:
            The sampling function. Every call creates a new sample according to the
            sampling scheme.
    """

    sample_options = NodeInput("Ping Pong", ntype=T.Selector(T.String(), ["Simple Rotation", "Complex Rotation", "Ping Pong","Random","Random Star"]))
    truncation = NodeInput(1.0, ntype=T.Float(0, 10))
    rotation_speed = NodeInput(0.125, ntype=T.Float(0, 1))
    sampler = NodeOutput(ntype=T.Sampler())

    def __init__(self, **kwargs):
        super(NoiseSampler, self).__init__(**kwargs)
        self.sampler = self.sample
        self.star_center = self.truncation
        self.step_counter = 1
        self.class_pool = list(range(0, 5))
        self.sample_schema = {
            "Ping Pong": NoiseSampler.PingPong,
            "Simple Rotation": NoiseSampler.SimpleRotation,
            "Complex Rotation": NoiseSampler.ComplexRotation,
            "Random": NoiseSampler.Random,
            "RandomStar": NoiseSampler.RandomStar,
        }
        self.sample_mode = self.switch_mode()
        self._in_edges["sample_options"].ntype.known_values = self.sample_schema.keys()
        self.inner_option = self.sample_options

    def update(self):
        if self.inner_option != self.sample_options:
            self.inner_option = self.sample_options
            self.sample_mode = self.switch_mode()

    def sample(self):
        return self.sample_mode.generate(self.rotation_speed) * self.truncation

    def switch_mode(self):
        return self.sample_schema[self.sample_options](self.device)

    def to_device(self, device):
        super().to_device(device)
        self.sample_mode = self.switch_mode()

    class PingPong:
        def __init__(self, device, **kwargs):
            self.vector = create_noise_vector(
                batch_size=1, dim_z=128, truncation=1, device=device,
            )

        def generate(self, rotation_speed):
            self.vector = -self.vector
            return self.vector

    class SimpleRotation:
        def __init__(self, device, **kwargs):
            self.vector = torch.zeros((1, 128), device=device)
            self.rotation = 0

        def rotate(self, rotation_speed):
            self.rotation += rotation_speed * math.pi
            self.vector[:, 0] = sin(self.rotation)
            self.vector[:, 1] = cos(self.rotation)

        def generate(self, rotation_speed):
            self.rotate(rotation_speed)
            return self.vector

    class ComplexRotation:
        def __init__(self, device):
            self.vector = torch.zeros((1, 128), device=device)
            self.params = torch.zeros((64, 2), device=device).normal_()
            self.rotation = 0

        def rotate(self, rotation_speed):
            self.rotation += rotation_speed * math.pi
            for i, params in enumerate(self.params):
                speed, distance = params
                self.vector[:, 2 * i + 0] = distance * sin(speed * self.rotation)
                self.vector[:, 2 * i + 1] = distance * cos(speed * self.rotation)

        def generate(self, rotation_speed):
            self.rotate(rotation_speed)
            return self.vector

    class Random:
        def __init__(self, device):
            self.device = device

        def generate(self, rotation_speed):
            return create_noise_vector(
                batch_size=1, dim_z=128, truncation=1, device=self.device,
            )

    class RandomStar:
        def __init__(self, device):
            self.device = device
            self.center = self.old_vector = (
                create_noise_vector(
                    batch_size=1, dim_z=128, truncation=1, device=self.device,
                )
                * 0.0
            )
            self.center_next = False

        def generate(self, rotation_speed):
            if self.center_next:
                vector = self.center
            else:
                vector = create_noise_vector(
                    batch_size=1, dim_z=128, truncation=1, device=self.device,
                )
            self.center_next = not self.center_next
            return vector


class SampleManager(Node):
    """
    Manages the sampling and correct replacement of some variable that can be sampled,
    such as noise or class.

    Inputs:
        time:
            Some time value. If it is lower than the value of the previous frame,
            it signals a beat.
        sampler:
            The sample function from which samples are drawn.

    Outputs:
        vector0:
            The current output vector.
        vector1:
            The next output vector.
    """

    time = NodeInput(ntype=T.Float())
    sampler = NodeInput(ntype=T.Sampler())

    vector0 = NodeOutput(ntype=T.Tensor())
    vector1 = NodeOutput(ntype=T.Tensor())

    def __init__(self, **kwargs):
        super(SampleManager, self).__init__(**kwargs)
        self.sample = iter(range(2))
        self.inner_vec0 = next(self.sample)
        self.inner_vec1 = next(self.sample)
        self.vector0 = self.inner_vec0
        self.vector1 = self.inner_vec1
        self.refresh = True
        self.internal_time = 0

    def update(self):
        if self.refresh:
            self.recalculate()
            self.refresh = False
        if self.internal_time - self.time > 0.001:
            self.inner_vec0 = self.inner_vec1
            self.inner_vec1 = self.sampler()
        self.vector0 = self.inner_vec0
        self.vector1 = self.inner_vec1
        self.internal_time = self.time

    def recalculate(self):
        self.inner_vec0 = self.sampler()
        self.inner_vec1 = self.sampler()
        self.vector0 = self.inner_vec0
        self.vector1 = self.inner_vec1

    def to_device(self, device):
        super().to_device(device)
        self.refresh = True


class ModelMatrix(Node):
    """
    Generates a translation matrix from all possible input variables.

    Inputs:
        position:
            A tuple of (x,y,z) position coordinates.
        rotation:
            A tuple of (x,y,z) rotation data. Not safe against gimbal lock!
        scale:
            A tuple of(x,y,z) scaling values.

    Outputs:
        model_matrix:
            A translation matrix output.
    """

    position = NodeInput((0.0, 0.0, 0.0), ntype=T.Vector3(-5.0, 5.0))
    rotation = NodeInput((0.0, 0.0, 0.0), ntype=T.Vector3(-180.0, 180.0))
    scale = NodeInput((1.0, 1.0, 1.0), ntype=T.Vector3(0.0, 5.0))

    model_matrix = NodeOutput(ntype=T.Tensor((4, 4)))

    def __init__(self, **kwargs):
        super(ModelMatrix, self).__init__(**kwargs)

    def update(self):
        model = correctly_rotated_model_matrix()
        glm.rotate(model, self.rotation[0], 1, 0, 0)
        glm.rotate(model, self.rotation[1], 0, 1, 0)
        glm.rotate(model, self.rotation[2], 0, 0, 1)
        glm.translate(model, self.position[0], self.position[1], self.position[2])
        glm.scale(model, self.scale[0], self.scale[1], self.scale[2])
        self.model_matrix = model


class ComplexFunction(Node):
    """
    A complex function with multiple possible inputs. Also allows slower playback by
    changing the single_beat_length.

    Inputs:
        single_beat_length:
            The length of a single beat. So if 1, the x value will change
            from 0 to 1 over the course of a single beat. If 4, it will be 4 times as slow.
        x:
            An input time value in the normalized_time format. While other values could
            be used, this is the de facto standard.
        y:
            An additional input value, can be bound to something if needed.
        z:
            An additional input value, can be bound to something if needed.
        abc:
            A tuple of 3 additional inputs.
        function:
            A function to compute.

    Outputs:
        result:
            The result of computing the function. after changing the beat_length.
        linear_result:
            The result of just changing the beat_length with single_beat_length.
    """

    single_beat_length = NodeInput(1.0, ntype=T.Base2Int(1, 512))
    x = NodeInput(1.0, ntype=T.Float(0, 1))
    y = NodeInput(1.0, ntype=T.Float(0, 1))
    z = NodeInput(1.0, ntype=T.Float(0, 1))
    abc = NodeInput((1.0, 1.0, 1.0), ntype=T.Vector3(0, 1))
    function = NodeInput(
        "x+y", ntype=T.Selector(T.MathFunction(), ["x", "y", "0", "1","0.5-cos(x*3*pi)","x**y*(a)+(1-a)",
                "x**y*(a+b*sin(c*x*pi))+(1-(a+b*sin(c*x*pi)))"])
    )

    result = NodeOutput(ntype=T.Float())
    linear_result = NodeOutput(ntype=T.Float())

    def __init__(self, presets=None, bounds=(0, 1), **kwargs):
        super(ComplexFunction, self).__init__(**kwargs)
        if presets:
            self._in_edges["function"].ntype.known_values += presets
        self.working_function = self.function
        self.bounds = bounds

    def update(self):
        sbl = max(0.01, self.single_beat_length)
        x = math.fmod(self.x, sbl) / sbl
        self.linear_result = x
        try:
            result = numexpr.evaluate(
                self.function,
                local_dict={
                    "x": x,
                    "y": self.y,
                    "z": self.z,
                    "a": self.abc[0],
                    "b": self.abc[1],
                    "c": self.abc[2],
                    "pi": math.pi,
                },
            )

        except (KeyError, SyntaxError):
            result = numexpr.evaluate(
                self.working_function,
                local_dict={
                    "x": x,
                    "y": self.y,
                    "z": self.z,
                    "a": self.abc[0],
                    "b": self.abc[1],
                    "c": self.abc[2],
                    "pi": math.pi,
                },
            )
        self.result = np.clip(result, self.bounds[0], self.bounds[1])


class ScalarToVector(Node):
    """
    Creates a vector2 from a scalar value

    """

    scalar = NodeInput(0.0)

    vector = NodeOutput(ntype=T.Vector2())

    def __init__(self, vector_size=2, **kwargs):
        super(ScalarToVector, self).__init__(**kwargs)
        self.vector_size = vector_size

    def update(self):
        self.vector = [self.scalar] * self.vector_size
