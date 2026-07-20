import torch
import gc
import json
from visualize.font_settings import FontSettings
from watermark.auto_watermark import AutoWatermark
from utils.transformers_config import TransformersConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from visualize.page_layout_settings import PageLayoutSettings
from visualize.data_for_visualization import DataForVisualization
from visualize.visualizer import DiscreteVisualizer, ContinuousVisualizer
from visualize.legend_settings import DiscreteLegendSettings, ContinuousLegendSettings
from visualize.color_scheme import ColorSchemeForDiscreteVisualization, ColorSchemeForContinuousVisualization
from IPython.display import display
from PIL import Image, ImageOps, ImageFont, ImageDraw
from utils.utils import load_config_file

def test_discreet_visualization():
    tokens = ["PREFIX", "PREFIX", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test"]
    flags = [-1, -1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    weights = [0, 0, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4]

    discreet_visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                            font_settings=FontSettings(),
                                            page_layout_settings=PageLayoutSettings(),
                                            legend_settings=DiscreteLegendSettings())
    img = discreet_visualizer.visualize(data=DataForVisualization(tokens, flags, weights),
                                     show_text=True, visualize_weight=True, display_legend=True)

    img.save(".visualization/test1.png")
    display(img)


def test_continuous_visualization():
    tokens = ["PREFIX", "PREFIX", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test", "Hello", "world", "this", "is", "a", "test"]
    values = [None, None, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4]
    weights = [0, 0, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4, 0.1, 0.5, 0.3, 0.8, 0.2, 0.4]

    continuous_visualizer = ContinuousVisualizer(color_scheme=ColorSchemeForContinuousVisualization(),
                                                    font_settings=FontSettings(),
                                                    page_layout_settings=PageLayoutSettings(),
                                                    legend_settings=ContinuousLegendSettings())
    img = continuous_visualizer.visualize(data=DataForVisualization(tokens, values, weights),
                                        show_text=False, visualize_weight=False, display_legend=True)

    img.save(".visualization/test2.png")
    display(img)


def get_data(algorithm_name):
    # Load data
    with open('dataset/c4/processed_c4.json', 'r') as f:
        lines = f.readlines()
    item = json.loads(lines[5])
    prompt = item['prompt']

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Transformers config
    transformers_config = TransformersConfig(model=AutoModelForCausalLM.from_pretrained("./models/opt-1.3b").to(device),
                                            tokenizer=AutoTokenizer.from_pretrained("./models/opt-1.3b"),
                                            vocab_size=50272,
                                            device=device,
                                            max_new_tokens=100,
                                            min_length=130,
                                            no_repeat_ngram_size=4)

    myWatermark = AutoWatermark.load(f'{algorithm_name}',
                                     #algorithm_config=f'config/{algorithm_name}.json',  # f'config/{args.algorithm}.json'     # load_config_file(f'config/{algorithm_name}.json')
                                     algorithm_config=load_config_file(f'config/{algorithm_name}.json'), 
                                     transformers_config=transformers_config)

    watermarked_text = myWatermark.generate_watermarked_text(prompt)
    unwatermarked_text = myWatermark.generate_unwatermarked_text(prompt)
    watermarked_data = myWatermark.get_data_for_visualization(watermarked_text)
    unwatermarked_data = myWatermark.get_data_for_visualization(unwatermarked_text)

    return watermarked_data, unwatermarked_data


def test_visualization_without_weight(algorithm_name, visualize_type='discrete'):
    # Validate input
    assert visualize_type in ['discrete', 'continuous']
    if visualize_type == 'discrete':
        assert algorithm_name in ['KGW', 'Unigram', 'SWEET', 'UPV', 'SIR', 'XSIR', 'EWD', 'SMOOTH','Rethinking_uni','Rethinking_gaosi']
    else:
        assert algorithm_name in ['EXP', 'EXPEdit']

    # Get data for visualization
    watermarked_data, unwatermarked_data = get_data(algorithm_name)

    # Init visualizer
    if visualize_type == 'discrete':
        visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                        font_settings=FontSettings(),
                                        page_layout_settings=PageLayoutSettings(),
                                        legend_settings=DiscreteLegendSettings())
    else:
        visualizer = ContinuousVisualizer(color_scheme=ColorSchemeForContinuousVisualization(),
                                        font_settings=FontSettings(),
                                        page_layout_settings=PageLayoutSettings(),
                                        legend_settings=ContinuousLegendSettings())

    # Visualize
    watermarked_img = visualizer.visualize(data=watermarked_data,
                                           show_text=True,
                                           visualize_weight=True,
                                           display_legend=True)

    unwatermarked_img = visualizer.visualize(data=unwatermarked_data,
                                             show_text=True,
                                             visualize_weight=True,
                                             display_legend=True)

    watermarked_img.save(f"{algorithm_name}_watermarked.pdf", format="pdf")  # dpi=(600, 600), 
    unwatermarked_img.save(f"{algorithm_name}_unwatermarked.pdf", format="pdf")

    watermarked_width, watermarked_height = watermarked_img.size
    unwatermarked_width, unwatermarked_height = unwatermarked_img.size

    font_size = 22
    font = ImageFont.truetype("./font/times.ttf", font_size)
    title_height = 80

    new_watermarked_img = Image.new('RGB', (watermarked_width, watermarked_height + title_height), (255, 255, 255))
    new_unwatermarked_img = Image.new('RGB', (unwatermarked_width, watermarked_height + title_height), (255, 255, 255))

    draw1 = ImageDraw.Draw(new_watermarked_img)
    text_bbox1 = draw1.textbbox((0, 0), f"{algorithm_name} watermarked", font=font)
    draw1.text((int((watermarked_width - text_bbox1[2] - text_bbox1[0]) * 0.3), int(title_height * 0.35)), f"{algorithm_name} watermarked", fill=(0, 0, 0), font=font)

    draw2 = ImageDraw.Draw(new_unwatermarked_img)
    text_bbox2 = draw2.textbbox((0, 0), f"{algorithm_name} unwatermarked", font=font)
    draw2.text((int((unwatermarked_width - text_bbox2[2] - text_bbox2[0]) * 0.3), int(title_height * 0.35)), f"{algorithm_name} unwatermarked", fill=(0, 0, 0), font=font)

    new_watermarked_img.paste(watermarked_img, (0, title_height))
    new_unwatermarked_img.paste(unwatermarked_img, (0, title_height))

    total_width = watermarked_width + unwatermarked_width
    max_height = watermarked_height + title_height

    new_img = Image.new('RGB', (total_width, max_height))

    new_img.paste(new_watermarked_img, (0, 0))
    new_img.paste(new_unwatermarked_img, (watermarked_width, 0))
    display(new_img)

def test_visualization_with_weight(algorithm_name):
    # Validate input
    assert algorithm_name in ['SWEET', 'EWD']

    # Get data for visualization
    watermarked_data, unwatermarked_data = get_data(algorithm_name)

    # Init visualizer
    visualizer = DiscreteVisualizer(color_scheme=ColorSchemeForDiscreteVisualization(),
                                    font_settings=FontSettings(),
                                    page_layout_settings=PageLayoutSettings(),
                                    legend_settings=DiscreteLegendSettings())

    # Visualize
    watermarked_img = visualizer.visualize(data=watermarked_data,
                                           show_text=True,
                                           visualize_weight=True,
                                           display_legend=True)

    unwatermarked_img = visualizer.visualize(data=unwatermarked_data,
                                             show_text=True,
                                             visualize_weight=True,
                                             display_legend=True)

    watermarked_img.save(f"{algorithm_name}_watermarked.png")
    unwatermarked_img.save(f"{algorithm_name}_unwatermarked.png")

    watermarked_width, watermarked_height = watermarked_img.size
    unwatermarked_width, unwatermarked_height = unwatermarked_img.size

    font_size = 22
    font = ImageFont.truetype("./font/times.ttf", font_size)
    title_height = 80

    new_watermarked_img = Image.new('RGB', (watermarked_width, watermarked_height + title_height), (255, 255, 255))
    new_unwatermarked_img = Image.new('RGB', (unwatermarked_width, watermarked_height + title_height), (255, 255, 255))

    draw1 = ImageDraw.Draw(new_watermarked_img)
    text_bbox1 = draw1.textbbox((0, 0), f"{algorithm_name} watermarked", font=font)
    draw1.text((int((watermarked_width - text_bbox1[2] - text_bbox1[0]) * 0.3), int(title_height * 0.35)), f"{algorithm_name} watermarked", fill=(0, 0, 0), font=font)

    draw2 = ImageDraw.Draw(new_unwatermarked_img)
    text_bbox2 = draw2.textbbox((0, 0), f"{algorithm_name} unwatermarked", font=font)
    draw2.text((int((unwatermarked_width - text_bbox2[2] - text_bbox2[0]) * 0.3), int(title_height * 0.35)), f"{algorithm_name} unwatermarked", fill=(0, 0, 0), font=font)

    new_watermarked_img.paste(watermarked_img, (0, title_height))
    new_unwatermarked_img.paste(unwatermarked_img, (0, title_height))

    total_width = watermarked_width + unwatermarked_width
    max_height = watermarked_height + title_height

    new_img = Image.new('RGB', (total_width, max_height))

    new_img.paste(new_watermarked_img, (0, 0))
    new_img.paste(new_unwatermarked_img, (watermarked_width, 0))
    display(new_img)
    
    
    
    
# test_discreet_visualization()

# test_continuous_visualization()

test_visualization_without_weight('EWD', 'discrete')

# test_visualization_without_weight('EXP', 'continuous')

# test_visualization_with_weight('SWEET')

# test_visualization_with_weight('EWD')