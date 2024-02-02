import os
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import cairosvg
from io import BytesIO
import defusedxml

Image.MAX_IMAGE_PIXELS = None # WARNING! Do not run this code without care of the downloaded database
defusedxml.common._apply_defusing = lambda x: x


def read_svg_with_pillow(svg_path):
    # Convert SVG to PNG using cairosvg
    svg_content = open(svg_path, 'rb').read()
    png_data = cairosvg.svg2png(file_obj=BytesIO(svg_content))

    # Read PNG data with Pillow
    pil_image = Image.open(BytesIO(png_data))

    return pil_image

def read_image_any_format(path):
    _, file_extension = os.path.splitext(path.lower())

    if file_extension == '.pdf':
        # Read the first page of the PDF using PyMuPDF and convert it to a PIL Image
        pdf_document = fitz.open(path)
        pdf_page = pdf_document[0]
        pdf_image = pdf_page.get_pixmap()
        pixels = np.frombuffer(pdf_image.samples, dtype=np.uint8)
        pdf_numpy_array = pixels.reshape((pdf_image.height, pdf_image.width,\
                                          int(pixels.shape[0] / (pdf_image.height * pdf_image.width))))
        pil_image = Image.fromarray(pdf_numpy_array)

    elif file_extension in ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp', '.tif']:
        # Read the image using PIL directly
        pil_image = Image.open(path)

    elif file_extension == '.svg':

        pil_image = read_svg_with_pillow(path)

    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

    return pil_image.convert('RGB')