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

#### Windows

```sh
pip install -r requirements.txt
```

#### macOS/Linux

```sh
pip3 install -r requirements.txt
```

### 4. Initialize Databases

#### Windows

```sh
python init_db.py
python init_face_db.py
```

#### macOS/Linux

```sh
python3 init_db.py
python3 init_face_db.py
```

## Running the Application

### Start the Web Interface

#### Windows

```sh
python app.py
```

#### macOS/Linux

```sh
python3 app.py
```

### Start the Person Tracking and Recognition System

#### Windows

```sh
python main9_4.py
```

#### macOS/Linux

```sh
python3 main9_4.py
```

## Features

- Multi-camera person tracking
- Face recognition using FaceNet
- YOLOv8-based person detection
- DeepSORT for tracking
- Flask web interface for real-time monitoring

##

