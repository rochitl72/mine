# mine
# Multi-Camera Person Tracking and Recognition System

## Installation and Setup

### 1. Clone the Repository

```sh
git clone https://github.com/rochitl72/mine.git
cd mine
```

### 2. Create Necessary Directories

```sh
mkdir templates
mv index.html templates/
```

### 3. Install Dependencies

```sh
pip install -r requirements.txt
```

### 4. Initialize Databases

```sh
python init_db.py
python init_face_db.py
```

## Running the Application

### Start the Web Interface

```sh
python app.py
```

### Start the Person Tracking and Recognition System

```sh
python main9_4.py
```

## Features

- Multi-camera person tracking
- Face recognition using FaceNet
- YOLOv8-based person detection
- DeepSORT for tracking
- Flask web interface for real-time monitoring

##

