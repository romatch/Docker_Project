import time
from pathlib import Path
from flask import Flask, request
from detect import run
import uuid
import yaml
from loguru import logger
import os
from botocore.exceptions import NoCredentialsError
import boto3
from pymongo import MongoClient

images_bucket = os.environ['BUCKET_NAME']

with open("data/coco128.yaml", "r") as stream:
    names = yaml.safe_load(stream)['names']

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    # Generates a UUID for this current prediction HTTP request. This id can be used as a reference in logs to
    # identify and track individual prediction requests.
    prediction_id = str(uuid.uuid4())

    logger.info(f'prediction: {prediction_id}. start processing')

    # Receives a URL parameter representing the image to download from S3
    img_name = request.args.get('imgName')

    # TODO download img_name from S3, store the local image path in original_img_path
    #  The bucket name should be provided as an env var BUCKET_NAME.
    # Initialize an S3 client
    s3 = boto3.client('s3')

    # Define the local directory where you want to store the downloaded image
    local_directory = '/home/romka/Docker_Project/yolo5/download_img/'  # You can specify your desired directory

    # Ensure the local directory exists; create it if it doesn't
    os.makedirs(local_directory, exist_ok=True)

    # Specify the local file path where the image will be stored
    original_img_path = os.path.join(local_directory, img_name)

    # Download the image from S3 to the local directory
    try:
        s3.download_file(images_bucket, img_name, original_img_path)
        logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')
    except NoCredentialsError:
        logger.error('AWS credentials not available. Ensure AWS credentials are properly configured.')
        return 'AWS credentials not available', 500

    logger.info(f'prediction: {prediction_id}/{original_img_path}. Download img completed')

    # Predicts the objects in the image
    run(
        weights='yolov5s.pt',
        data='data/coco128.yaml',
        source=original_img_path,
        project='static/data',
        name=prediction_id,
        save_txt=True
    )

    logger.info(f'prediction: {prediction_id}/{original_img_path}. done')

    # This is the path for the predicted image with labels The predicted image typically includes bounding boxes
    # drawn around the detected objects, along with class labels and possibly confidence scores.
    predicted_img_path = Path(f'static/data/{prediction_id}/{original_img_path}')

    # TODO Uploads the predicted image (predicted_img_path) to S3 (be careful not to override the original image).
    # Define the S3 key (path) where you want to upload the predicted image
    s3_key = f'predicted/{prediction_id}/{img_name}'  # Adjust the key as needed

    try:
        # Upload the predicted image to S3
        s3.upload_file(predicted_img_path, images_bucket, s3_key)
        logger.info(f'prediction: {prediction_id}/{original_img_path}. Predicted image uploaded to S3')
    except NoCredentialsError:
        logger.error('AWS credentials not available. Ensure AWS credentials are properly configured.')
        return 'AWS credentials not available', 500

    # Parse prediction labels and create a summary
    pred_summary_path = Path(f'static/data/{prediction_id}/labels/{original_img_path.split(".")[0]}.txt')
    if pred_summary_path.exists():
        with open(pred_summary_path) as f:
            labels = f.read().splitlines()
            labels = [line.split(' ') for line in labels]
            labels = [{
                'class': names[int(l[0])],
                'cx': float(l[1]),
                'cy': float(l[2]),
                'width': float(l[3]),
                'height': float(l[4]),
            } for l in labels]

        logger.info(f'prediction: {prediction_id}/{original_img_path}. prediction summary:\n\n{labels}')

        prediction_summary = {
            'prediction_id': prediction_id,
            'original_img_path': original_img_path,
            'predicted_img_path': predicted_img_path,
            'labels': labels,
            'time': time.time()
        }

        # TODO store the prediction_summary in MongoDB
        mongo_client = MongoClient('mongodb://localhost:27017/')

        # Choose a database and collection name
        db = mongo_client['yolo5_predictions']
        collection = db['image_predictions']

        # Insert the prediction_summary into the MongoDB collection
        collection.insert_one(prediction_summary)
        logger.info(f'prediction: {prediction_id}/{original_img_path}. Prediction summary stored in MongoDB')

        return prediction_summary
    else:
        return f'prediction: {prediction_id}/{original_img_path}. prediction result not found', 404


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8081)
