import telebot
from loguru import logger
import os
import time
from telebot.types import InputFile
from img_proc import Img
import boto3
import requests
import json
import sys


class Bot:
    def __init__(self, token, telegram_chat_url):
        # create a new instance of the TeleBot class.
        # all communication with Telegram servers are done using self.telegram_bot_client
        self.telegram_bot_client = telebot.TeleBot(token)
        # remove any existing webhooks configured in Telegram servers
        self.telegram_bot_client.remove_webhook()
        time.sleep(0.5)
        # set the webhook URL
        self.telegram_bot_client.set_webhook(url=f'{telegram_chat_url}/{token}/', timeout=60)
        logger.info(f'Telegram Bot information\n\n{self.telegram_bot_client.get_me()}')

    def send_text(self, chat_id, text):
        self.telegram_bot_client.send_message(chat_id, text)

    def send_text_with_quote(self, chat_id, text, quoted_msg_id):
        self.telegram_bot_client.send_message(chat_id, text, reply_to_message_id=quoted_msg_id)

    def is_current_msg_photo(self, msg):
        return 'photo' in msg

    def download_user_photo(self, msg):
        """
        Downloads the photos that sent to the Bot to `photos` directory (should be existed)
        :return:
        """
        if not self.is_current_msg_photo(msg):
            raise RuntimeError(f'Message content of type \'photo\' expected')

        file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        data = self.telegram_bot_client.download_file(file_info.file_path)
        folder_name = file_info.file_path.split('/')[0]

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        with open(file_info.file_path, 'wb') as photo:
            photo.write(data)

        return file_info.file_path

    def send_photo(self, chat_id, img_path):
        if not os.path.exists(img_path):
            raise RuntimeError("Image path doesn't exist")

        self.telegram_bot_client.send_photo(
            chat_id,
            InputFile(img_path)
        )

    def handle_message(self, msg):
        """Bot Main message handler"""
        logger.info(f'Incoming message: {msg}')
        if 'text' in msg:
            self.send_text(msg['chat']['id'], f'We upload picture here,but did you say?: {msg["text"]}')


class QuoteBot(Bot):
    def handle_message(self, msg):
        logger.info(f'Incoming message: {msg}')
        if msg["text"] != 'Please don\'t quote me':
            self.send_text_with_quote(msg['chat']['id'], msg["text"], quoted_msg_id=msg["message_id"])


class ImageProcessingBot(Bot):
    def __init__(self, token, telegram_chat_url):
        super().__init__(token, telegram_chat_url)
        self.processing_completed = True

    def handle_message(self, msg):
        if not self.processing_completed:
            logger.info("Previous message processing is not completed. Ignoring current message.")
            return

        if "photo" in msg:
            # If the message contains a photo, check if it also has a caption
            if "caption" in msg:
                caption = msg["caption"]
                if "contour" in caption.lower():
                    self.process_image_contur(msg)
                if "rotate" in caption.lower():
                    self.process_image_rotate(msg)
                if "predict" in caption.lower():
                    self.upload_and_predict(msg)
                if "blur" in caption.lower():
                    self.process_image_blur(msg)
            else:
                logger.info("Received photo without a caption.")
        elif "text" in msg:
            super().handle_message(msg)  # Call the parent class method to handle text messages

    def process_image(self, msg):
        self.processing_completed = False

        # Download the two photos sent by the user
        image_path = self.download_user_photo(msg)
        another_image_path = self.download_user_photo(msg)

        # Create two different Img objects from the downloaded images
        image = Img(image_path)
        # Save the processed image to the specified folder
        processed_image_path = image.save_img()

        if processed_image_path is not None:
            # Send the processed image back to the user
            self.send_photo(msg['chat']['id'], processed_image_path)

    def process_image_contur(self, msg):
        self.processing_completed = False
        self.send_text(msg['chat']['id'], text=f'A few moments later =)')
        # Download the two photos sent by the user
        image_path = self.download_user_photo(msg)

        # Create two different Img objects from the downloaded images
        image = Img(image_path)

        # Process the image using your custom methods (e.g., apply filter)
        image.contour()  # contur the image

        # Save the processed image to the specified folder
        processed_image_path = image.save_img()

        if processed_image_path is not None:
            # Send the processed image back to the user
            self.send_text(msg['chat']['id'], text=f'Done!\nHere you go:')
            self.send_photo(msg['chat']['id'], processed_image_path)

        self.processing_completed = True

    def process_image_rotate(self, msg):
        self.processing_completed = False
        self.send_text(msg['chat']['id'], text=f'A few moments later =)')
        # Download the two photos sent by the user
        image_path = self.download_user_photo(msg)

        # Create two different Img objects from the downloaded images
        image = Img(image_path)

        # Process the image using your custom methods (e.g., apply filter)
        image.rotate()  # rotate the image

        # Save the processed image to the specified folder
        processed_image_path = image.save_img()

        if processed_image_path is not None:
            # Send the processed image back to the user
            self.send_text(msg['chat']['id'], text=f'Done!\nHere you go:')
            self.send_photo(msg['chat']['id'], processed_image_path)

        self.processing_completed = True

    def upload_and_predict(self, msg):
        self.processing_completed = False
        self.send_text(msg['chat']['id'], text=f'A few moments later =)')
        # Download the photo sent by the user
        # file_info = self.telegram_bot_client.get_file(msg['photo'][-1]['file_id'])
        # file_path_parts = file_info.file_path.split('/')
        # file_name = file_path_parts[-1]
        image_path = self.download_user_photo(msg)
        # Upload the image to S3
        s3_client = boto3.client('s3')
        images_bucket = 'romans-s3-bucket'
        s3_key = f'{msg["chat"]["id"]}.jpeg'
        s3_client.upload_file(image_path, images_bucket, s3_key)

        time.sleep(5)

        # Send a request to the YOLO5 microservice # with the containers name once its build
        yolo5_url = f'http://yolo5-app:8081/predict?imgName={s3_key}'
        response = requests.post(yolo5_url)
        if response.status_code == 200:
            # Print the JSON response as text
            json_response = response.text
            print(json_response)
            sys.stdout.flush()

            # Parse the Json file and send user a message:
            response_data = json.loads(json_response)
            # Initialize a dictionary to store the class counts
            class_counts = {}

            # Iterate through the labels and count the occurrences of each class
            for label in response_data['labels']:
                class_name = label['class']
                if class_name in class_counts:
                    class_counts[class_name] += 1
                else:
                    class_counts[class_name] = 1

            # Create a message with the detected objects and their counts
            message = "This what I find:\n"
            for class_name, count in class_counts.items():
                message += f"{class_name}: {count}\n"

            # Send the message to the user
            self.telegram_bot_client.send_message(msg['chat']['id'], message)

        self.processing_completed = True

    def process_image_blur(self, msg):
        self.processing_completed = False
        self.send_text(msg['chat']['id'], text=f'A few moments later =)')

        # Download the two photos sent by the user
        image_path = self.download_user_photo(msg)

        # Create two different Img objects from the downloaded images
        image = Img(image_path)

        # Process the image using your custom methods (e.g., apply filter)
        image.blur()  # Blurs the image

        # Save the processed image to the specified folder
        processed_image_path = image.save_img()

        if processed_image_path is not None:
            # Send the processed image back to the user
            self.send_text(msg['chat']['id'], text=f'Done!\nHere you go:')
            self.send_photo(msg['chat']['id'], processed_image_path)

        self.processing_completed = True
