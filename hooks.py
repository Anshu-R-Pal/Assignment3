# utils/activation_hooks.py
activation_memory = {}

def activation_hook(name):
    """
    Returns a forward hook that records the number of elements and shape
    of the activation tensor under the given name.
    """
    def hook(module, input, output):
        try:
            nelems = int(output.numel())
            activation_memory[name] = {
                "num_elements": nelems,
                "shape": tuple(output.shape)
            }
        except Exception:
            # not a tensor or cannot measure - skip
            pass
    return hook


