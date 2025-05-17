FROM public.ecr.aws/lambda/python:3.11

# Install Python packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy your function code
COPY lambda_function.py .
COPY preprocessing.py .
COPY aws_functions.py .
COPY model_training_prediction.py .
# Set the Lambda handler
CMD ["lambda_function.lambda_handler"]
