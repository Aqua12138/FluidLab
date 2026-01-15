from fluidlab.utils.misc import is_headless
if not is_headless():
    from .ggui_renderer import GGUIRenderer
    # Import GLRenderer only when needed to avoid compilation issues
    # GLRenderer will be imported lazily when GL renderer type is selected
    try:
        from .gl_renderer import GLRenderer
    except ImportError:
        GLRenderer = None