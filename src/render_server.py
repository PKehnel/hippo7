from threading import Thread

from glumpy import app
from torch.cuda import is_available

from renderer.graph_setup import complex_gan_setup  # noqa
from renderer.graph_setup import debug_graph  # noqa
from renderer.graph_setup import no_gan_setup  # noqa
from renderer.graph_setup import simple_gan_setup
from renderer.graph_setup import visualize_graph  # noqa
from renderer.opengl.render import WindowManager
from renderer.server import start_server

model_type = "biggan-deep-512"
fullscreen = False


def get_device():
    return "cuda:0" if is_available() else "cpu"


if get_device().startswith("cuda"):  # This optimizes performance in cuda mode
    print("CUDA enabled, using GPU!")
    from torch.backends import cudnn  # QA

    if cudnn.is_available():
        cudnn.enabled = True
        cudnn.benchmark = True

# create window with OpenGL context
app.use("pyglet")
window = app.Window(512, 512, fullscreen=False, decoration=True)
WindowManager.bind_window(window)

# Instantiate generator
# generator = simple_gan_setup(model_type=model_type)
# generator = no_gan_setup()
generator = complex_gan_setup(model_type=model_type)
visualize_graph(generator)
generator.to_device(get_device())


@window.event
def on_draw(dt):
    window.set_title(str(window.fps))
    generator.draw()


@window.event
def on_resize(width, height):
    WindowManager.resize(width, height)


@window.event
def on_key_release(symbol, modifiers):
    global fullscreen
    if symbol == 102:
        fullscreen = not fullscreen
        window.set_fullscreen(fullscreen)
    print("Key released (symbol=%s, modifiers=%s)" % (symbol, modifiers))


t = Thread(target=start_server, args=[generator, ("localhost", 50036)], daemon=True)
t.start()

app.run()

t.join()
