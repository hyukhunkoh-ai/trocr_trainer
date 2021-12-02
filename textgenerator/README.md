# TextRecognitionDataGenerator [![TravisCI](https://travis-ci.org/Belval/TextRecognitionDataGenerator.svg?branch=master)](https://travis-ci.org/Belval/TextRecognitionDataGenerator) [![PyPI version](https://badge.fury.io/py/trdg.svg)](https://badge.fury.io/py/trdg) [![codecov](https://codecov.io/gh/Belval/TextRecognitionDataGenerator/branch/master/graph/badge.svg)](https://codecov.io/gh/Belval/TextRecognitionDataGenerator) [![Documentation Status](https://readthedocs.org/projects/textrecognitiondatagenerator/badge/?version=latest)](https://textrecognitiondatagenerator.readthedocs.io/en/latest/?badge=latest) [![mattermost](https://img.shields.io/badge/help-mattermost-blue)](https://mattermost.belval.org/signup_user_complete/?id=6j1pj6itd7y4781o1u813796ry)

### Text skewing

What if you want random skewing? Add `-k` and `-rk` (`trdg -c 1000 -w 5 -f 64 -k 5 -rk`)

![6](samples/6.jpg "6")
![7](samples/7.jpg "7")
![8](samples/8.jpg "8")
![9](samples/9.jpg "9")
![10](samples/10.jpg "10")

### Text distortion
You can also add distorsion to the generated text with `-d` and `-do`

![23](samples/24.jpg "0")
![24](samples/25.jpg "1")
![25](samples/26.jpg "2")

### Text blurring

But scanned document usually aren't that clear are they? Add `-bl` and `-rbl` to get gaussian blur on the generated image with user-defined radius (here 0, 1, 2, 4):

![11](samples/11.jpg "0")
![12](samples/12.jpg "1")
![13](samples/13.jpg "2")
![14](samples/14.jpg "4")

### Background

Maybe you want another background? Add `-b` to define one of the three available backgrounds: gaussian noise (0), plain white (1), quasicrystal (2) or image (3).

![15](samples/15.jpg "0")
![16](samples/16.jpg "1")
![17](samples/17.jpg "2")
![23](samples/23.jpg "3")

When using image background (3). A image from the images/ folder will be randomly selected and the text will be written on it.

### Handwritten

Or maybe you are working on an OCR for handwritten text? Add `-hw`! (Experimental)

![18](samples/18.jpg "0")
![19](samples/19.jpg "1")
![20](samples/20.jpg "2")
![21](samples/21.jpg "3")
![22](samples/22.jpg "4")

It uses a Tensorflow model trained using [this excellent project](https://github.com/Grzego/handwriting-generation) by Grzego.

**The project does not require TensorFlow to run if you aren't using this feature**

### Dictionary

The text is chosen at random in a dictionary file (that can be found in the *dicts* folder) and drawn on a white background made with Gaussian noise. The resulting image is saved as [text]\_[index].jpg

There are a lot of parameters that you can tune to get the results you want, therefore I recommend checking out `trdg -h` for more information.

## Create images with Chinese text

It is simple! Just do `trdg -l cn -c 1000 -w 5`!

Generated texts come both in simplified and traditional Chinese scripts.

Traditional:

![27](samples/27.jpg "0")

Simplified:

![28](samples/28.jpg "1")

## Create images with Japanese text 

It is simple! Just do `trdg -l ja -c 1000 -w 5`!

Output 

![29](samples/29.jpg "2")


## Add new fonts

The script picks a font at random from the *fonts* directory.

| Directory | Languages |
|:----|:-----|
| fonts/latin | English, French, Spanish, German |
| fonts/cn | Chinese |
| fonts/ko | Korean |
| fonts/ja | Japanese |

Simply add/remove fonts until you get the desired output.

If you want to add a new non-latin language, the amount of work is minimal.

1. Create a new folder with your language [two-letters code](https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes)
2. Add a .ttf font in it
3. Edit `run.py` to add an if statement in `load_fonts()`
4. Add a text file in `dicts` with the same two-letters code
5. Run the tool as you normally would but add `-l` with your two-letters code

It only supports .ttf for now.
