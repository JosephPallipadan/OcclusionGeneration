import base64
from selenium import webdriver
import pdb
import time
from os import listdir, remove
from shutil import copyfile

image_directory = "images"

chrome_driver_path = "C:/WebDriver/bin/chromedriver.exe"  # Replace with chrome driver path

# Edit this string with the path of the jeelizGlassesVTOWidget/index.html
index_file_path = "C:/Users/Joseph/Documents/PA/Sunglasses Occlusion Generation/jeelizGlassesVTOWidget/index.html"

time_between_clicks = 1  # Seconds
time_between_images = 3  # Seconds

driver = webdriver.Chrome(executable_path=chrome_driver_path)
driver.get(index_file_path)
time.sleep(10) # Sleep for 10 seconds to wait for webcam permission to be allowed

canvas = driver.find_element_by_id("JeelizVTOWidgetCanvas")
transparentGlassesButton = driver.find_element_by_id('transparentGlasses')
changeGlassesButton = driver.find_element_by_id('changeGlasses')
loadingWidget = driver.find_element_by_id('JeelizVTOWidgetLoading')


def extractOccludedImage(filename: str):
    # get the canvas as a PNG base64 string
    canvas_base64 = driver.execute_script(
        "return arguments[0].toDataURL('image/png').substring(21);", canvas)

    # decode
    canvas_png = base64.b64decode(canvas_base64)

    # save
    with open(f'results/{filename}.png', 'wb') as f:
        f.write(canvas_png)


def setTransparentGlasses():
    transparentGlassesButton.click()
    while '_jeelizVTOForceHide' not in loadingWidget.get_attribute('class'):
        True
    time.sleep(time_between_clicks)


def changeGlasses():
    changeGlassesButton.click()
    while '_jeelizVTOForceHide' not in loadingWidget.get_attribute('class'):
        True
    time.sleep(time_between_clicks)


def collect_images():
    for file in listdir(image_directory):
        copyfile(f'{image_directory}/{file}', f'to-occlude.jpg')
        time.sleep(time_between_images)
        file = file[:-4]

        def extractGroundTruth():
            print(f"Extracting Ground Truth Image: {file}")
            setTransparentGlasses()
            extractOccludedImage(f"{file}-ground-truth")
            print(f"Done Extracting Ground Truth Image: {file} \n")

        def extractOccluded(mark):
            print(f"Extracting Occluded Image: {file}-{mark}")
            changeGlasses()
            extractOccludedImage(f"{file}-{mark}")
            print(f"Done Extracting Occluded Image: {file}-{mark}\n")

        extractGroundTruth()
        extractOccluded('A')
        extractOccluded('B')
        extractOccluded('C')

        remove(f'to-occlude.jpg')


collect_images()
