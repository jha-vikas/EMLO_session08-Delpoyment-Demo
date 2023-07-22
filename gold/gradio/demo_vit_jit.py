from typing import List, Tuple
from PIL import Image
import torch
from torchvision import transforms as T
from torch.nn import functional as F

import hydra
import gradio as gr
from omegaconf import DictConfig

from gold import utils

log = utils.get_pylogger(__name__)

def demo(cfg: DictConfig) -> Tuple[dict, dict]:
    """Demo function.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    with open(cfg.labels_path, 'r') as f:
        catgs = f.read()
    catgs = catgs.strip().splitlines()
    catgs = [i.strip().lower() for i in catgs]

    log.info("Running Demo")

    log.info(f"Instantiating scripted model <{cfg.ckpt_path}>")
    model = torch.jit.load(cfg.ckpt_path)

    log.info(f"Loaded Model: {model}")

    predict_transform = T.Compose(
                            [
                                T.Resize((32, 32)), 
                                T.ToTensor(),
                                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ]
                        )


    def recognize_img(image: PIL.Image):
        if image is None:
            return None
        
        image = predict_transform(image)
        image = torch.unsqueeze(image[0], 0)
        #image = torch.tensor(image[None, None, ...], dtype=torch.float32)
        logits = model.forward(image)
        
        preds = F.softmax(logits, dim=1).squeeze(0).tolist()

        out = torch.topk(torch.tensor(preds), len(catgs))
        topk_prob  = out[0].tolist()
        topk_label = out[1].tolist()
        
        confidence_map = {catgs[topk_label[i]]: topk_prob[i] for i in range(len(catgs))}
        print(confidence_map)
        return confidence_map

    #im = gr.Image(shape=(32, 32), image_mode="L", invert_colors=True, source="canvas")

    demo = gr.Interface(
        fn=recognize_img,
        inputs=gr.Image(type="pil"),
        outputs=[gr.Label(num_top_classes=10)],
        live=True,
    )

    demo.launch(server_name= "0.0.0.0", server_port=5000, share=True)

@hydra.main(
    version_base="1.2", config_path="../../configs", config_name="infer_vit_jit.yaml"
)
def main(cfg: DictConfig) -> None:
    demo(cfg)

if __name__ == "__main__":
    main()