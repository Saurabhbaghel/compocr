#from google.colab.patches import cv2_imshow
import os
from fastapi import UploadFile
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
# from argparse import ArgumentParser
import cv2
import pytesseract
import shutil
try:
 from PIL import Image
except ImportError:
 import Image
import layoutparser as lp
import json



def detection(img):
  # os.environ['USE_TORCH'] = '1'
  doc = img
  predictor = ocr_predictor(pretrained=True)
  result = predictor(doc)
  json_export = result.export()
  return json_export

def recognition(lang_model, image):
  tessdata_dir_config = r'--tessdata-dir "indic-parser/configs/tessdata"'
  os.environ["TESSDATA_PREFIX"] = 'indic-parser/configs/tessdata'
  languages=pytesseract.get_languages(config=tessdata_dir_config)

  if lang_model in languages:
    ocr_agent = lp.TesseractAgent(languages=lang_model)

  res = ocr_agent.detect(image, return_response = True)
  tesseract_output = res["data"].to_dict('list')
  # task = convert_to_ls(image, tesseract_output, file_name, per_level='block_num')
  return tesseract_output

 
def hocr(data):

  header = '''
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
  <head>
    <title></title>
    <meta http-equiv="Content-Type" content="text/html;charset=utf-8"/>
    <meta name='ocr-system' content='tesseract v5.0.1.20220118' />
    <meta name='ocr-capabilities' content='ocr_page ocr_line'/>
  </head>
  <body>
    <div class='ocr_page' id='page_1'>
'''
  lines = []
  for i, result in enumerate(data['predictions'][0]['result']):
    img_name = result['img_name']
    bbox = result['bbox']
    bbox_str = f"{bbox['x1']} {bbox['y1']} {bbox['x2']} {bbox['y2']}"
    text = result['text']
    if text:
      lines.append(f"\n     <span class='ocr_line' id='line_1_{i+1}' title='bbox {bbox_str}'>{text}</span>")

  for line in lines:
    header = header + line
  footers = ['\n    </div>\n',' </body>\n','</html>\n']
  for footer in footers:
    header = header + footer

  return header

 
def main(args: UploadFile):
  os.environ['USE_TORCH'] = '1'
  image_dir = "testing/"
  image = args.file.read() #.path_img
  # image = cv2.imread(img_name)
  image = DocumentFile.from_images(image)
  json_export = detection(image)
  h, w = image[0].shape[:-1]
  i=0
  if os.path.exists(image_dir):
    shutil.rmtree(image_dir)
  os.mkdir(image_dir)
  output = []
  for block in json_export['pages'][0]['blocks']:
    for line in  block['lines']:
      i+=1
      p1,p2 = line['geometry']
      x1,y1 = int(p1[0]*w), int(p1[1]*h)
      x2,y1 = int(p2[0]*w), y1
      x2,y2 = x2, int(p2[1]*h)
      x1,y2 = x1, y2
      p1=(x1,y1)
      p2=(x2,y2)

      file_name = f'{image_dir}{i}.jpg'
      file = image[0][y1:y2,x1:x2,:]
      cv2.imwrite(file_name,file)
      img=Image.fromarray(file)
      tesseract_output = recognition("san_iitb", img)
      text = "".join(tesseract_output['text'])
      # print(text)
      name = f'{i}.jpg'
      output.append([name, [x1,y1,x2,y2], text])
  
  results = []
  for k in output:
    bbox = {
        'x1':k[1][0],
        'y1':k[1][1],
        'x2':k[1][2],
        'y2':k[1][3],
    }
    img_data = {
      'img_name': k[0],
      'bbox': bbox,
      'text': k[2]
    }
    results.extend([img_data])

  
  json_file = {
      'predictions': [{
          'result': results
      }]
  }
  
  hocr_data = hocr(json_file)
  
  with open("output.json", "w") as outfile:
    json.dump(json_file, outfile, ensure_ascii=False, indent=4)

  shutil.rmtree(image_dir)

  return hocr_data



# def arg_parser():
# 	parser = ArgumentParser()
# 	parser.add_argument('--path_img',help='address of the image file')
# 	args = parser.parse_args()
# 	return args
#
#
# if __name__ == '__main__':
#   args=arg_parser()
#   main(args)
