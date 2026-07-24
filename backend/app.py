from flask import (
    Flask,
    Response,
    jsonify,
    request,
    send_from_directory,
)

from flask_cors import CORS

from datetime import (
    date,
    datetime,
    time,
    timezone,
)

import csv
import io
import os
import re
import threading
import traceback
import uuid

import cv2
import mediapipe as mp
import numpy as np

from azure.core.exceptions import (
    ResourceExistsError,
    ResourceNotFoundError,
)

from azure.storage.blob import (
    BlobServiceClient,
    ContentSettings,
)

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    text,
)

from sqlalchemy.engine import URL

from sqlalchemy.exc import (
    IntegrityError,
    SQLAlchemyError,
)


# ---------------------------------------------------------
# Flask configuration
# ---------------------------------------------------------
app = Flask(__name__)

CORS(app)


BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    BACKEND_DIR
)

FRONTEND_DIR = os.path.join(
    PROJECT_ROOT,
    "frontend",
)


# ---------------------------------------------------------
# PostgreSQL configuration
# ---------------------------------------------------------
def build_database_url():
    database_url = os.environ.get(
        "DATABASE_URL"
    )

    if database_url:
        if database_url.startswith(
            "postgres://"
        ):
            database_url = (
                database_url.replace(
                    "postgres://",
                    "postgresql+psycopg://",
                    1,
                )
            )

        elif database_url.startswith(
            "postgresql://"
        ):
            database_url = (
                database_url.replace(
                    "postgresql://",
                    "postgresql+psycopg://",
                    1,
                )
            )

        return database_url

    required_variables = [
        "DB_HOST",
        "DB_NAME",
        "DB_USER",
        "DB_PASSWORD",
    ]

    missing_variables = [
        variable
        for variable in required_variables
        if not os.environ.get(variable)
    ]

    if missing_variables:
        raise RuntimeError(
            (
                "Missing database environment "
                "variables: "
                + ", ".join(
                    missing_variables
                )
            )
        )

    return URL.create(
        drivername=(
            "postgresql+psycopg"
        ),

        username=os.environ.get(
            "DB_USER"
        ),

        password=os.environ.get(
            "DB_PASSWORD"
        ),

        host=os.environ.get(
            "DB_HOST"
        ),

        port=int(
            os.environ.get(
                "DB_PORT",
                "5432",
            )
        ),

        database=os.environ.get(
            "DB_NAME"
        ),

        query={
            "sslmode": os.environ.get(
                "DB_SSLMODE",
                "require",
            ),
        },
    )


engine = create_engine(
    build_database_url(),

    pool_pre_ping=True,

    pool_recycle=300,
)


# ---------------------------------------------------------
# Azure Blob Storage configuration
# ---------------------------------------------------------
AZURE_STORAGE_CONNECTION_STRING = (
    os.environ.get(
        "AZURE_STORAGE_CONNECTION_STRING"
    )
)

AZURE_STORAGE_CONTAINER = (
    os.environ.get(
        "AZURE_STORAGE_CONTAINER",
        "face-images",
    )
)


if not AZURE_STORAGE_CONNECTION_STRING:
    raise RuntimeError(
        (
            "AZURE_STORAGE_CONNECTION_STRING "
            "is missing"
        )
    )


blob_service_client = (
    BlobServiceClient
    .from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )
)


container_client = (
    blob_service_client
    .get_container_client(
        AZURE_STORAGE_CONTAINER
    )
)


try:
    container_client.create_container()

except ResourceExistsError:
    pass


# ---------------------------------------------------------
# Database schema
# ---------------------------------------------------------
metadata = MetaData()


students_table = Table(
    "students",
    metadata,

    Column(
        "student_id",
        String(100),
        primary_key=True,
    ),

    Column(
        "student_name",
        String(255),
        nullable=False,
    ),

    Column(
        "image_path",
        Text,
        nullable=False,
    ),
)


student_images_table = Table(
    "student_images",
    metadata,

    Column(
        "id",
        Integer,
        primary_key=True,
    ),

    Column(
        "student_id",
        String(100),

        ForeignKey(
            "students.student_id",
            ondelete="CASCADE",
        ),

        nullable=False,
    ),

    Column(
        "image_path",
        Text,
        nullable=False,
    ),

    Column(
        "appearance_label",
        String(255),
        nullable=False,
    ),

    Column(
        "created_at",
        DateTime(
            timezone=True
        ),

        nullable=False,

        server_default=text(
            "CURRENT_TIMESTAMP"
        ),
    ),
)


classes_table = Table(
    "classes",
    metadata,

    Column(
        "id",
        Integer,
        primary_key=True,
    ),

    Column(
        "course_name",
        String(255),
        nullable=False,
    ),

    Column(
        "course_code",
        String(100),
        nullable=False,
        unique=True,
    ),

    Column(
        "professor_name",
        String(255),
        nullable=False,
    ),
)


class_students_table = Table(
    "class_students",
    metadata,

    Column(
        "class_id",
        Integer,

        ForeignKey(
            "classes.id",
            ondelete="CASCADE",
        ),

        primary_key=True,
    ),

    Column(
        "student_id",
        String(100),

        ForeignKey(
            "students.student_id",
            ondelete="CASCADE",
        ),

        primary_key=True,
    ),
)


attendance_table = Table(
    "attendance",
    metadata,

    Column(
        "id",
        Integer,
        primary_key=True,
    ),

    Column(
        "class_id",
        Integer,

        ForeignKey(
            "classes.id",
            ondelete="CASCADE",
        ),

        nullable=False,
    ),

    Column(
        "student_id",
        String(100),

        ForeignKey(
            "students.student_id",
            ondelete="CASCADE",
        ),

        nullable=False,
    ),

    Column(
        "student_name",
        String(255),
        nullable=False,
    ),

    Column(
        "timestamp",
        DateTime(
            timezone=True
        ),

        nullable=False,

        server_default=text(
            "CURRENT_TIMESTAMP"
        ),
    ),

    Column(
        "status",
        String(50),
        nullable=False,

        server_default=text(
            "'Present'"
        ),
    ),

    Column(
        "attendance_photo_path",
        Text,
        nullable=True,
    ),

    Column(
        "recognition_distance",
        Float,
        nullable=True,
    ),
)


attendance_policies_table = Table(
    "attendance_policies",
    metadata,

    Column(
        "id",
        Integer,
        primary_key=True,
    ),

    Column(
        "class_id",
        Integer,

        ForeignKey(
            "classes.id",
            ondelete="CASCADE",
        ),

        nullable=False,
    ),

    Column(
        "policy_name",
        String(255),
        nullable=False,
    ),

    Column(
        "absence_limit",
        Integer,
        nullable=True,
    ),

    Column(
        "late_limit",
        Integer,
        nullable=True,
    ),

    Column(
        "late_minutes",
        Integer,
        nullable=True,
    ),

    Column(
        "attendance_weight",
        Float,
        nullable=True,
    ),

    Column(
        "consequence",
        Text,
        nullable=True,
    ),

    Column(
        "excuse_counts",
        String(20),
        nullable=False,

        server_default=text(
            "'No'"
        ),
    ),
)


excuses_table = Table(
    "excuses",
    metadata,

    Column(
        "id",
        Integer,
        primary_key=True,
    ),

    Column(
        "student_id",
        String(100),

        ForeignKey(
            "students.student_id",
            ondelete="CASCADE",
        ),

        nullable=False,
    ),

    Column(
        "class_id",
        Integer,

        ForeignKey(
            "classes.id",
            ondelete="CASCADE",
        ),

        nullable=False,
    ),

    Column(
        "excuse_date",
        Date,
        nullable=False,
    ),

    Column(
        "reason",
        Text,
        nullable=False,
    ),

    Column(
        "status",
        String(50),
        nullable=False,

        server_default=text(
            "'Pending'"
        ),
    ),

    Column(
        "submitted_at",
        DateTime(
            timezone=True
        ),

        nullable=False,

        server_default=text(
            "CURRENT_TIMESTAMP"
        ),
    ),
)


metadata.create_all(
    engine
)


# ---------------------------------------------------------
# Safe schema migration
# ---------------------------------------------------------
with engine.begin() as connection:
    connection.execute(
        text(
            """
            ALTER TABLE attendance
            ADD COLUMN IF NOT EXISTS
                attendance_photo_path TEXT
            """
        )
    )

    connection.execute(
        text(
            """
            ALTER TABLE attendance
            ADD COLUMN IF NOT EXISTS
                recognition_distance
                DOUBLE PRECISION
            """
        )
    )


# ---------------------------------------------------------
# Lazy-loaded AI models
# ---------------------------------------------------------
_deepface = None

_ai_lock = threading.Lock()


def get_deepface():
    global _deepface

    if _deepface is None:
        with _ai_lock:
            if _deepface is None:
                from deepface import (
                    DeepFace
                )

                _deepface = DeepFace

    return _deepface


# ---------------------------------------------------------
# General helper functions
# ---------------------------------------------------------
def safe_remove(file_path):
    if not file_path:
        return

    try:
        if os.path.exists(file_path):
            os.remove(file_path)

    except OSError as error:
        print(
            (
                "Could not remove file "
                f"{file_path}: {error}"
            ),
            flush=True,
        )


def parse_date(value):
    try:
        return date.fromisoformat(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None


def optional_int(value):
    if (
        value is None
        or str(value).strip() == ""
    ):
        return None

    return int(value)


def optional_float(value):
    if (
        value is None
        or str(value).strip() == ""
    ):
        return None

    return float(value)


def serialize_value(value):
    if isinstance(
        value,
        (
            datetime,
            date,
        ),
    ):
        return value.isoformat()

    return value


def serialize_row(row):
    return {
        key: serialize_value(value)
        for key, value in row.items()
    }


def sanitize_label(value):
    safe_value = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(value).strip(),
    )

    return (
        safe_value.strip("_")
        or "appearance"
    )


# ---------------------------------------------------------
# Azure Blob helper functions
# ---------------------------------------------------------
def upload_face_image(
    image_file,
    student_id,
    appearance_label,
):
    original_filename = (
        os.path.basename(
            image_file.filename
            or "image.jpg"
        )
    )

    extension = (
        os.path.splitext(
            original_filename
        )[1].lower()
        or ".jpg"
    )

    blob_name = (
        f"students/"
        f"{sanitize_label(student_id)}/"
        f"{sanitize_label(appearance_label)}/"
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    image_file.stream.seek(0)

    blob_client = (
        container_client
        .get_blob_client(
            blob_name
        )
    )

    blob_client.upload_blob(
        image_file.stream,

        overwrite=True,

        content_settings=(
            ContentSettings(
                content_type=(
                    image_file.content_type
                    or "image/jpeg"
                )
            )
        ),
    )

    return blob_name


def upload_attendance_photo(
    image_bytes,
    class_id,
    student_id,
):
    current_time = datetime.now(
        timezone.utc
    )

    blob_name = (
        f"attendance/"
        f"{class_id}/"
        f"{sanitize_label(student_id)}/"
        f"{current_time.date().isoformat()}/"
        f"{uuid.uuid4().hex}.jpg"
    )

    blob_client = (
        container_client
        .get_blob_client(
            blob_name
        )
    )

    blob_client.upload_blob(
        image_bytes,

        overwrite=False,

        content_settings=(
            ContentSettings(
                content_type="image/jpeg"
            )
        ),
    )

    return blob_name


def delete_blob_if_exists(
    blob_name,
):
    if not blob_name:
        return

    try:
        container_client.delete_blob(
            blob_name
        )

    except ResourceNotFoundError:
        pass

    except Exception as error:
        print(
            (
                "Could not delete Azure blob "
                f"{blob_name}: {error}"
            ),
            flush=True,
        )


def download_blob_as_image(
    blob_name,
):
    blob_client = (
        container_client
        .get_blob_client(
            blob_name
        )
    )

    blob_bytes = (
        blob_client
        .download_blob()
        .readall()
    )

    image_array = np.frombuffer(
        blob_bytes,
        np.uint8,
    )

    image = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            (
                "Could not decode stored "
                f"image: {blob_name}"
            )
        )

    return image


# ---------------------------------------------------------
# Face extraction
# ---------------------------------------------------------
def extract_face_with_mediapipe(
    image,
):
    if (
        image is None
        or image.size == 0
    ):
        return (
            None,
            "Image could not be read",
        )

    image_height, image_width = (
        image.shape[:2]
    )

    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    )

    with (
        mp.solutions
        .face_detection
        .FaceDetection(
            model_selection=1,

            min_detection_confidence=(
                0.60
            ),
        )
    ) as face_detector:
        results = (
            face_detector.process(
                rgb_image
            )
        )

    detections = (
        results.detections
        if results
        and results.detections
        else []
    )

    if len(detections) == 0:
        return (
            None,
            (
                "No face detected in "
                "uploaded image"
            ),
        )

    if len(detections) > 1:
        return (
            None,
            (
                "Multiple faces detected. "
                "Please capture one face only"
            ),
        )

    bounding_box = (
        detections[0]
        .location_data
        .relative_bounding_box
    )

    x = int(
        bounding_box.xmin
        * image_width
    )

    y = int(
        bounding_box.ymin
        * image_height
    )

    width = int(
        bounding_box.width
        * image_width
    )

    height = int(
        bounding_box.height
        * image_height
    )

    if width <= 0 or height <= 0:
        return (
            None,
            "Invalid face detected",
        )

    horizontal_padding = int(
        width * 0.25
    )

    vertical_padding = int(
        height * 0.30
    )

    x1 = max(
        0,
        x - horizontal_padding,
    )

    y1 = max(
        0,
        y - vertical_padding,
    )

    x2 = min(
        image_width,
        (
            x
            + width
            + horizontal_padding
        ),
    )

    y2 = min(
        image_height,
        (
            y
            + height
            + vertical_padding
        ),
    )

    face_crop = image[
        y1:y2,
        x1:x2,
    ]

    if (
        face_crop is None
        or face_crop.size == 0
    ):
        return (
            None,
            (
                "Face crop could not "
                "be created"
            ),
        )

    face_height, face_width = (
        face_crop.shape[:2]
    )

    if (
        face_width < 80
        or face_height < 80
    ):
        return (
            None,
            (
                "Face is too small or "
                "too far away"
            ),
        )

    return (
        face_crop,
        None,
    )


# ---------------------------------------------------------
# Basic API status routes
# ---------------------------------------------------------

@app.route(
    "/api",
    methods=["GET"],
)
def api_status():
    return jsonify(
        {
            "success": True,
            "message": (
                "Smart Attendance API "
                "is running"
            ),
        }
    ), 200
@app.route("/debug-versions", methods=["GET"])
def debug_versions():
    import google.protobuf
    import tensorflow as tf
    return jsonify({
        "protobuf": google.protobuf.__version__,
        "tensorflow": tf.__version__,
    }), 200

@app.route(
    "/health",
    methods=["GET"],
)
def health_check():
    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        return jsonify(
            {
                "success": True,
                "status": "healthy",
                "database": "connected",
                "storage": "configured",
            }
        ), 200

    except Exception as error:
        print(
            (
                "Health check error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "status": "unhealthy",
                "error": str(error),
            }
        ), 500
# ---------------------------------------------------------
# Face detection
# ---------------------------------------------------------
@app.route(
    "/detect",
    methods=["POST"],
)
def detect_face():
    uploaded_image = request.files.get(
        "image"
    )

    if not uploaded_image:
        return jsonify(
            {
                "success": False,
                "error": "Image is required",
            }
        ), 400

    try:
        image_bytes = (
            uploaded_image.read()
        )

        image_array = np.frombuffer(
            image_bytes,
            np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Uploaded image could "
                        "not be decoded"
                    ),
                }
            ), 400

        face_crop, face_error = (
            extract_face_with_mediapipe(
                image
            )
        )

        if face_error:
            return jsonify(
                {
                    "success": False,
                    "face_detected": False,
                    "error": face_error,
                }
            ), 400

        face_height, face_width = (
            face_crop.shape[:2]
        )

        return jsonify(
            {
                "success": True,
                "face_detected": True,
                "face_width": face_width,
                "face_height": face_height,
            }
        ), 200

    except Exception as error:
        print(
            (
                "Face detection error: "
                f"{repr(error)}"
            ),
            flush=True,
        )

        traceback.print_exc()

        return jsonify(
            {
                "success": False,
                "error": (
                    "Face detection failed"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Student registration
# ---------------------------------------------------------
@app.route(
    "/register",
    methods=["POST"],
)
def register_student():
    student_name = (
        request.form.get(
            "student_name",
            ""
        ).strip()
    )

    student_id = (
        request.form.get(
            "student_id",
            ""
        ).strip()
    )

    uploaded_image = request.files.get(
        "image"
    )

    if (
        not student_name
        or not student_id
        or not uploaded_image
    ):
        return jsonify(
            {
                "success": False,
                "error": (
                    "Student name, student ID, "
                    "and image are required"
                ),
            }
        ), 400

    uploaded_blob_name = None

    try:
        image_bytes = (
            uploaded_image.read()
        )

        image_array = np.frombuffer(
            image_bytes,
            np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Uploaded image could "
                        "not be decoded"
                    ),
                }
            ), 400

        _, face_error = (
            extract_face_with_mediapipe(
                image
            )
        )

        if face_error:
            return jsonify(
                {
                    "success": False,
                    "error": face_error,
                }
            ), 400

        uploaded_image.stream.seek(0)

        uploaded_blob_name = (
            upload_face_image(
                uploaded_image,
                student_id,
                "primary",
            )
        )

        with engine.begin() as connection:
            existing_student = (
                connection.execute(
                    text(
                        """
                        SELECT student_id
                        FROM students
                        WHERE student_id =
                              :student_id
                        """
                    ),
                    {
                        "student_id": student_id,
                    },
                ).first()
            )

            if existing_student:
                delete_blob_if_exists(
                    uploaded_blob_name
                )

                return jsonify(
                    {
                        "success": False,
                        "error": (
                            "A student with this "
                            "ID already exists"
                        ),
                    }
                ), 409

            connection.execute(
                text(
                    """
                    INSERT INTO students
                        (
                            student_id,
                            student_name,
                            image_path
                        )
                    VALUES
                        (
                            :student_id,
                            :student_name,
                            :image_path
                        )
                    """
                ),
                {
                    "student_id": student_id,
                    "student_name": (
                        student_name
                    ),
                    "image_path": (
                        uploaded_blob_name
                    ),
                },
            )

            connection.execute(
                text(
                    """
                    INSERT INTO student_images
                        (
                            student_id,
                            image_path,
                            appearance_label
                        )
                    VALUES
                        (
                            :student_id,
                            :image_path,
                            :appearance_label
                        )
                    """
                ),
                {
                    "student_id": student_id,
                    "image_path": (
                        uploaded_blob_name
                    ),
                    "appearance_label": (
                        "Primary"
                    ),
                },
            )

        return jsonify(
            {
                "success": True,
                "message": (
                    "Student registered "
                    "successfully"
                ),
                "student_id": student_id,
                "student_name": student_name,
            }
        ), 201

    except IntegrityError as error:
        if uploaded_blob_name:
            delete_blob_if_exists(
                uploaded_blob_name
            )

        print(
            (
                "Registration integrity "
                f"error: {error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Student could not be "
                    "registered because the "
                    "record already exists"
                ),
            }
        ), 409

    except Exception as error:
        if uploaded_blob_name:
            delete_blob_if_exists(
                uploaded_blob_name
            )

        print(
            (
                "Registration error: "
                f"{repr(error)}"
            ),
            flush=True,
        )

        traceback.print_exc()

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not register student"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Alternate appearance
# ---------------------------------------------------------
@app.route(
    "/add-appearance",
    methods=["POST"],
)
def add_appearance():
    student_id = (
        request.form.get(
            "student_id",
            ""
        ).strip()
    )

    appearance_label = (
        request.form.get(
            "appearance_label",
            ""
        ).strip()
    )

    uploaded_image = request.files.get(
        "image"
    )

    if (
        not student_id
        or not appearance_label
        or not uploaded_image
    ):
        return jsonify(
            {
                "success": False,
                "error": (
                    "Student ID, appearance "
                    "label, and image are "
                    "required"
                ),
            }
        ), 400

    uploaded_blob_name = None

    try:
        with engine.connect() as connection:
            student_exists = (
                connection.execute(
                    text(
                        """
                        SELECT student_id
                        FROM students
                        WHERE student_id =
                              :student_id
                        """
                    ),
                    {
                        "student_id": student_id,
                    },
                ).first()
            )

        if not student_exists:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Student not found"
                    ),
                }
            ), 404

        image_bytes = (
            uploaded_image.read()
        )

        image_array = np.frombuffer(
            image_bytes,
            np.uint8,
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Uploaded image could "
                        "not be decoded"
                    ),
                }
            ), 400

        _, face_error = (
            extract_face_with_mediapipe(
                image
            )
        )

        if face_error:
            return jsonify(
                {
                    "success": False,
                    "error": face_error,
                }
            ), 400

        uploaded_image.stream.seek(0)

        uploaded_blob_name = (
            upload_face_image(
                uploaded_image,
                student_id,
                appearance_label,
            )
        )

        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO student_images
                        (
                            student_id,
                            image_path,
                            appearance_label
                        )
                    VALUES
                        (
                            :student_id,
                            :image_path,
                            :appearance_label
                        )
                    """
                ),
                {
                    "student_id": student_id,
                    "image_path": (
                        uploaded_blob_name
                    ),
                    "appearance_label": (
                        appearance_label
                    ),
                },
            )

        return jsonify(
            {
                "success": True,
                "message": (
                    "Alternate appearance "
                    "added successfully"
                ),
                "student_id": student_id,
                "appearance_label": (
                    appearance_label
                ),
            }
        ), 201

    except Exception as error:
        if uploaded_blob_name:
            delete_blob_if_exists(
                uploaded_blob_name
            )

        print(
            (
                "Add appearance error: "
                f"{repr(error)}"
            ),
            flush=True,
        )

        traceback.print_exc()

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not add alternate "
                    "appearance"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Students
# ---------------------------------------------------------
@app.route(
    "/students",
    methods=["GET"],
)
def get_students():
    try:
        with engine.connect() as connection:
            students = (
                connection.execute(
                    text(
                        """
                        SELECT
                            students.student_id,
                            students.student_name,
                            students.image_path,
                            COUNT(
                                student_images.id
                            ) AS appearance_count
                        FROM students
                        LEFT JOIN student_images
                            ON students.student_id =
                               student_images.student_id
                        GROUP BY
                            students.student_id,
                            students.student_name,
                            students.image_path
                        ORDER BY
                            students.student_name ASC
                        """
                    )
                ).mappings().all()
            )

        return jsonify(
            {
                "success": True,
                "students": [
                    serialize_row(student)
                    for student in students
                ],
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            (
                "Get students error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not load students"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Classes
# ---------------------------------------------------------
@app.route(
    "/classes",
    methods=["POST"],
)
def create_class():
    data = (
        request.get_json(silent=True)
        or request.form
    )

    course_name = (
        str(
            data.get(
                "course_name",
                ""
            )
        ).strip()
    )

    course_code = (
        str(
            data.get(
                "course_code",
                ""
            )
        ).strip()
    )

    professor_name = (
        str(
            data.get(
                "professor_name",
                ""
            )
        ).strip()
    )

    if (
        not course_name
        or not course_code
        or not professor_name
    ):
        return jsonify(
            {
                "success": False,
                "error": (
                    "Course name, course code, "
                    "and professor name are "
                    "required"
                ),
            }
        ), 400

    try:
        with engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    INSERT INTO classes
                        (
                            course_name,
                            course_code,
                            professor_name
                        )
                    VALUES
                        (
                            :course_name,
                            :course_code,
                            :professor_name
                        )
                    RETURNING id
                    """
                ),
                {
                    "course_name": (
                        course_name
                    ),
                    "course_code": (
                        course_code
                    ),
                    "professor_name": (
                        professor_name
                    ),
                },
            )

            class_id = result.scalar_one()

        return jsonify(
            {
                "success": True,
                "class_id": class_id,
                "course_name": course_name,
                "course_code": course_code,
                "professor_name": (
                    professor_name
                ),
            }
        ), 201

    except IntegrityError:
        return jsonify(
            {
                "success": False,
                "error": (
                    "A class with this course "
                    "code already exists"
                ),
            }
        ), 409

    except SQLAlchemyError as error:
        print(
            (
                "Create class error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not create class"
                ),
            }
        ), 500


@app.route(
    "/classes",
    methods=["GET"],
)
def get_classes():
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        classes.id,
                        classes.course_name,
                        classes.course_code,
                        classes.professor_name,
                        COUNT(
                            class_students.student_id
                        ) AS student_count
                    FROM classes
                    LEFT JOIN class_students
                        ON classes.id =
                           class_students.class_id
                    GROUP BY
                        classes.id,
                        classes.course_name,
                        classes.course_code,
                        classes.professor_name
                    ORDER BY
                        classes.course_code ASC
                    """
                )
            ).mappings().all()

        return jsonify(
            {
                "success": True,
                "classes": [
                    serialize_row(row)
                    for row in rows
                ],
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            (
                "Get classes error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not load classes"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Class enrollment
# ---------------------------------------------------------
@app.route(
    "/class-students",
    methods=["POST"],
)
def add_student_to_class():
    data = (
        request.get_json(silent=True)
        or request.form
    )

    class_id = data.get(
        "class_id"
    )

    student_id = (
        str(
            data.get(
                "student_id",
                ""
            )
        ).strip()
    )

    if not class_id or not student_id:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Class ID and student ID "
                    "are required"
                ),
            }
        ), 400

    try:
        class_id = int(class_id)

    except (
        TypeError,
        ValueError,
    ):
        return jsonify(
            {
                "success": False,
                "error": "Invalid class ID",
            }
        ), 400

    try:
        with engine.begin() as connection:
            class_exists = (
                connection.execute(
                    text(
                        """
                        SELECT id
                        FROM classes
                        WHERE id = :class_id
                        """
                    ),
                    {
                        "class_id": class_id,
                    },
                ).first()
            )

            if not class_exists:
                return jsonify(
                    {
                        "success": False,
                        "error": (
                            "Class not found"
                        ),
                    }
                ), 404

            student_exists = (
                connection.execute(
                    text(
                        """
                        SELECT student_id
                        FROM students
                        WHERE student_id =
                              :student_id
                        """
                    ),
                    {
                        "student_id": student_id,
                    },
                ).first()
            )

            if not student_exists:
                return jsonify(
                    {
                        "success": False,
                        "error": (
                            "Student not found"
                        ),
                    }
                ), 404

            connection.execute(
                text(
                    """
                    INSERT INTO class_students
                        (
                            class_id,
                            student_id
                        )
                    VALUES
                        (
                            :class_id,
                            :student_id
                        )
                    """
                ),
                {
                    "class_id": class_id,
                    "student_id": student_id,
                },
            )

        return jsonify(
            {
                "success": True,
                "message": (
                    "Student added to class"
                ),
            }
        ), 201

    except IntegrityError:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Student is already enrolled "
                    "in this class"
                ),
            }
        ), 409

    except SQLAlchemyError as error:
        print(
            (
                "Add class student error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not add student "
                    "to class"
                ),
            }
        ), 500


@app.route(
    "/class-students/<int:class_id>",
    methods=["GET"],
)
def get_class_students(class_id):
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        students.student_id,
                        students.student_name,
                        students.image_path
                    FROM class_students
                    JOIN students
                        ON class_students.student_id =
                           students.student_id
                    WHERE class_students.class_id =
                          :class_id
                    ORDER BY
                        students.student_name ASC
                    """
                ),
                {
                    "class_id": class_id,
                },
            ).mappings().all()

        return jsonify(
            {
                "success": True,
                "students": [
                    serialize_row(row)
                    for row in rows
                ],
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            (
                "Get class students error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not load class "
                    "students"
                ),
            }
        ), 500


@app.route(
    (
        "/class-students/"
        "<int:class_id>/"
        "<string:student_id>"
    ),
    methods=["DELETE"],
)
def remove_student_from_class(
    class_id,
    student_id,
):
    try:
        with engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    DELETE FROM class_students
                    WHERE class_id = :class_id
                      AND student_id =
                          :student_id
                    """
                ),
                {
                    "class_id": class_id,
                    "student_id": student_id,
                },
            )

        if result.rowcount == 0:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Enrollment not found"
                    ),
                }
            ), 404

        return jsonify(
            {
                "success": True,
                "deleted": True,
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            (
                "Remove class student error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not remove student "
                    "from class"
                ),
            }
        ), 500
# ---------------------------------------------------------
# Face recognition and attendance marking
# ---------------------------------------------------------
@app.route(
    "/recognize",
    methods=["POST"],
)
def recognize_student():
    class_id = request.form.get(
        "class_id"
    )

    uploaded_image = request.files.get(
        "image"
    )

    if not class_id or not uploaded_image:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Class ID and captured "
                    "image are required"
                ),
            }
        ), 400

    try:
        class_id = int(class_id)

    except (
        TypeError,
        ValueError,
    ):
        return jsonify(
            {
                "success": False,
                "error": "Invalid class ID",
            }
        ), 400

    new_attendance_blob = None
    old_attendance_blob = None

    try:
        # -------------------------------------------------
        # Read and decode the webcam image
        # -------------------------------------------------
        uploaded_bytes = (
            uploaded_image.read()
        )

        if not uploaded_bytes:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Captured image is empty"
                    ),
                }
            ), 400

        uploaded_array = np.frombuffer(
            uploaded_bytes,
            np.uint8,
        )

        test_image = cv2.imdecode(
            uploaded_array,
            cv2.IMREAD_COLOR,
        )

        if test_image is None:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Captured image could "
                        "not be decoded"
                    ),
                }
            ), 400

        # -------------------------------------------------
        # Detect and crop the uploaded face
        # -------------------------------------------------
        test_face, face_error = (
            extract_face_with_mediapipe(
                test_image
            )
        )

        if face_error:
            return jsonify(
                {
                    "success": False,
                    "error": face_error,
                }
            ), 400

        # -------------------------------------------------
        # Confirm that the selected class exists
        # -------------------------------------------------
        with engine.connect() as connection:
            selected_class = (
                connection.execute(
                    text(
                        """
                        SELECT
                            id,
                            course_code,
                            course_name
                        FROM classes
                        WHERE id = :class_id
                        """
                    ),
                    {
                        "class_id": class_id,
                    },
                ).mappings().first()
            )

        if not selected_class:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Selected class "
                        "was not found"
                    ),
                }
            ), 404

        # -------------------------------------------------
        # Load every appearance belonging to students
        # enrolled in the selected class
        # -------------------------------------------------
        with engine.connect() as connection:
            appearance_rows = (
                connection.execute(
                    text(
                        """
                        SELECT
                            students.student_id,
                            students.student_name,
                            student_images.image_path,
                            student_images.appearance_label
                        FROM class_students
                        JOIN students
                            ON class_students.student_id =
                               students.student_id
                        JOIN student_images
                            ON students.student_id =
                               student_images.student_id
                        WHERE class_students.class_id =
                              :class_id

                        UNION ALL

                        SELECT
                            students.student_id,
                            students.student_name,
                            students.image_path,
                            'Primary'
                                AS appearance_label
                        FROM class_students
                        JOIN students
                            ON class_students.student_id =
                               students.student_id
                        WHERE class_students.class_id =
                              :class_id
                          AND students.image_path
                              IS NOT NULL
                          AND NOT EXISTS
                              (
                                  SELECT 1
                                  FROM student_images
                                  WHERE
                                      student_images.student_id =
                                      students.student_id
                                    AND
                                      student_images.image_path =
                                      students.image_path
                              )

                        ORDER BY
                            student_name,
                            student_id
                        """
                    ),
                    {
                        "class_id": class_id,
                    },
                ).mappings().all()
            )

        if not appearance_rows:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "No students with stored "
                        "face images are enrolled "
                        "in this class"
                    ),
                }
            ), 404

        deepface = get_deepface()

        match_threshold = 0.45

        student_best_matches = {}

        # -------------------------------------------------
        # Compare the captured face with every stored
        # appearance and keep only the best appearance
        # for each student.
        # -------------------------------------------------
        for appearance in appearance_rows:
            stored_blob_name = (
                appearance["image_path"]
            )

            if not stored_blob_name:
                continue

            try:
                stored_image = (
                    download_blob_as_image(
                        stored_blob_name
                    )
                )

                stored_face, stored_error = (
                    extract_face_with_mediapipe(
                        stored_image
                    )
                )

                if stored_error:
                    print(
                        (
                            "Skipping stored image "
                            f"{stored_blob_name}: "
                            f"{stored_error}"
                        ),
                        flush=True,
                    )

                    continue

                verification_result = (
                    deepface.verify(
                        img1_path=test_face,
                        img2_path=stored_face,

                        model_name=(
                            "VGG-Face"
                        ),

                        detector_backend=(
                            "skip"
                        ),

                        distance_metric=(
                            "cosine"
                        ),

                        enforce_detection=False,

                        align=False,

                        silent=True,
                    )
                )

                distance = float(
                    verification_result.get(
                        "distance",
                        float("inf"),
                    )
                )

                student_id = (
                    appearance["student_id"]
                )

                print(
                    (
                        "Compared against "
                        f"{student_id} "
                        f"({appearance['appearance_label']}): "
                        f"distance={distance:.6f}"
                    ),
                    flush=True,
                )

                current_best = (
                    student_best_matches.get(
                        student_id
                    )
                )

                if (
                    current_best is None
                    or distance
                    < current_best["distance"]
                ):
                    student_best_matches[
                        student_id
                    ] = {
                        "student_id": (
                            student_id
                        ),

                        "student_name": (
                            appearance[
                                "student_name"
                            ]
                        ),

                        "distance": (
                            distance
                        ),

                        "appearance_label": (
                            appearance[
                                "appearance_label"
                            ]
                        ),

                        "image_path": (
                            stored_blob_name
                        ),
                    }

            except Exception as comparison_error:
                print(
                    (
                        "Could not compare stored "
                        f"appearance "
                        f"{stored_blob_name}: "
                        f"{repr(comparison_error)}"
                    ),
                    flush=True,
                )

                traceback.print_exc()

                continue

        # -------------------------------------------------
        # Sort students by their best appearance distance
        # -------------------------------------------------
        ranked_matches = sorted(
            student_best_matches.values(),
            key=lambda match: (
                match["distance"]
            ),
        )

        if not ranked_matches:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "No valid stored face images "
                        "could be compared"
                    ),
                }
            ), 404

        best_match = ranked_matches[0]

        best_distance = (
            best_match["distance"]
        )

        second_best_distance = None

        if len(ranked_matches) > 1:
            second_best_distance = (
                ranked_matches[1][
                    "distance"
                ]
            )

        print(
            (
                "Best match: "
                f"{best_match['student_id']}, "
                f"distance={best_distance:.6f}"
            ),
            flush=True,
        )

        if second_best_distance is not None:
            print(
                (
                    "Second-best distance: "
                    f"{second_best_distance:.6f}"
                ),
                flush=True,
            )

        # -------------------------------------------------
        # Reject when even the closest match is too far
        # -------------------------------------------------
        if best_distance > match_threshold:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "No matching student found "
                        "in selected class"
                    ),
                    "best_distance": round(
                        best_distance,
                        4,
                    ),
                    "threshold": (
                        match_threshold
                    ),
                }
            ), 404

        # -------------------------------------------------
        # Reject when two students are almost equally close
        # -------------------------------------------------
        minimum_distance_gap = 0.03

        if (
            second_best_distance is not None
            and (
                second_best_distance
                - best_distance
            ) < minimum_distance_gap
        ):
            print(
                (
                    "Recognition rejected because "
                    "the two closest students were "
                    "too similar"
                ),
                flush=True,
            )

            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Face match was uncertain. "
                        "Please use a clearer photo"
                    ),
                    "best_distance": round(
                        best_distance,
                        4,
                    ),
                    "second_best_distance": round(
                        second_best_distance,
                        4,
                    ),
                }
            ), 409

        student_name = (
            best_match["student_name"]
        )

        student_id = (
            best_match["student_id"]
        )

        # -------------------------------------------------
        # Normalize the captured photo to JPEG before
        # uploading it to Azure Blob Storage.
        # -------------------------------------------------
        encoding_success, jpeg_buffer = (
            cv2.imencode(
                ".jpg",
                test_image,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    92,
                ],
            )
        )

        if not encoding_success:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Attendance photo could "
                        "not be prepared"
                    ),
                }
            ), 500

        attendance_photo_bytes = (
            jpeg_buffer.tobytes()
        )

        new_attendance_blob = (
            upload_attendance_photo(
                attendance_photo_bytes,
                class_id,
                student_id,
            )
        )

        # -------------------------------------------------
        # Insert or update today's attendance record
        # -------------------------------------------------
        current_time = datetime.now(
            timezone.utc
        )

        attendance_date = (
            current_time.date()
        )

        attendance_id = None
        attendance_was_updated = False

        try:
            with engine.begin() as connection:
                existing_record = (
                    connection.execute(
                        text(
                            """
                            SELECT
                                id,
                                attendance_photo_path
                            FROM attendance
                            WHERE class_id =
                                  :class_id
                              AND student_id =
                                  :student_id
                              AND CAST(
                                  timestamp AS DATE
                              ) = :attendance_date
                            LIMIT 1
                            """
                        ),
                        {
                            "class_id": (
                                class_id
                            ),
                            "student_id": (
                                student_id
                            ),
                            "attendance_date": (
                                attendance_date
                            ),
                        },
                    ).mappings().first()
                )

                if existing_record:
                    attendance_id = (
                        existing_record["id"]
                    )

                    old_attendance_blob = (
                        existing_record[
                            "attendance_photo_path"
                        ]
                    )

                    connection.execute(
                        text(
                            """
                            UPDATE attendance
                            SET
                                student_name =
                                    :student_name,
                                timestamp =
                                    :timestamp,
                                status =
                                    'Present',
                                attendance_photo_path =
                                    :attendance_photo_path,
                                recognition_distance =
                                    :recognition_distance
                            WHERE id =
                                  :attendance_id
                            """
                        ),
                        {
                            "student_name": (
                                student_name
                            ),

                            "timestamp": (
                                current_time
                            ),

                            "attendance_photo_path": (
                                new_attendance_blob
                            ),

                            "recognition_distance": (
                                best_distance
                            ),

                            "attendance_id": (
                                attendance_id
                            ),
                        },
                    )

                    attendance_was_updated = True

                else:
                    attendance_id = (
                        connection.execute(
                            text(
                                """
                                INSERT INTO attendance
                                    (
                                        class_id,
                                        student_id,
                                        student_name,
                                        timestamp,
                                        status,
                                        attendance_photo_path,
                                        recognition_distance
                                    )
                                VALUES
                                    (
                                        :class_id,
                                        :student_id,
                                        :student_name,
                                        :timestamp,
                                        :status,
                                        :attendance_photo_path,
                                        :recognition_distance
                                    )
                                RETURNING id
                                """
                            ),
                            {
                                "class_id": (
                                    class_id
                                ),

                                "student_id": (
                                    student_id
                                ),

                                "student_name": (
                                    student_name
                                ),

                                "timestamp": (
                                    current_time
                                ),

                                "status": (
                                    "Present"
                                ),

                                "attendance_photo_path": (
                                    new_attendance_blob
                                ),

                                "recognition_distance": (
                                    best_distance
                                ),
                            },
                        ).scalar_one()
                    )

        except Exception:
            # The database update failed, so remove the
            # newly uploaded attendance image.
            delete_blob_if_exists(
                new_attendance_blob
            )

            new_attendance_blob = None

            raise

        # Delete the previous attendance photo only after
        # the new database transaction completed.
        if (
            old_attendance_blob
            and old_attendance_blob
            != new_attendance_blob
        ):
            delete_blob_if_exists(
                old_attendance_blob
            )

        return jsonify(
            {
                "success": True,

                "student_name": (
                    student_name
                ),

                "student_id": (
                    student_id
                ),

                "class_id": (
                    class_id
                ),

                "attendance_id": (
                    attendance_id
                ),

                "attendance_marked": True,

                "attendance_updated": (
                    attendance_was_updated
                ),

                "status": "Present",

                "distance": round(
                    best_distance,
                    4,
                ),

                "threshold": (
                    match_threshold
                ),

                "matched_appearance": (
                    best_match[
                        "appearance_label"
                    ]
                ),

                "attendance_photo_available": (
                    True
                ),

                "attendance_photo_url": (
                    f"/attendance/"
                    f"{attendance_id}/photo"
                ),

                "timestamp": (
                    current_time.isoformat()
                ),
            }
        ), 200

    except ResourceNotFoundError as error:
        if new_attendance_blob:
            delete_blob_if_exists(
                new_attendance_blob
            )

        print(
            (
                "Recognition storage error: "
                f"{repr(error)}"
            ),
            flush=True,
        )

        traceback.print_exc()

        return jsonify(
            {
                "success": False,
                "error": (
                    "A stored face image could "
                    "not be loaded"
                ),
            }
        ), 500

    except Exception as error:
        if new_attendance_blob:
            # This is safe even when the blob was already
            # removed during database error handling.
            delete_blob_if_exists(
                new_attendance_blob
            )

        print(
            (
                "Recognition route error: "
                f"{repr(error)}"
            ),
            flush=True,
        )

        traceback.print_exc()

        return jsonify(
            {
                "success": False,
                "error": (
                    "Face recognition failed"
                ),
            }
        ), 500


# ---------------------------------------------------------
# View a captured attendance photo
# ---------------------------------------------------------
@app.route(
    "/attendance/<int:attendance_id>/photo",
    methods=["GET"],
)
def get_attendance_photo(
    attendance_id,
):
    try:
        with engine.connect() as connection:
            attendance_record = (
                connection.execute(
                    text(
                        """
                        SELECT
                            attendance_photo_path
                        FROM attendance
                        WHERE id =
                              :attendance_id
                        """
                    ),
                    {
                        "attendance_id": (
                            attendance_id
                        ),
                    },
                ).mappings().first()
            )

        if not attendance_record:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Attendance record "
                        "not found"
                    ),
                }
            ), 404

        blob_name = (
            attendance_record[
                "attendance_photo_path"
            ]
        )

        if not blob_name:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "No captured photo exists "
                        "for this attendance record"
                    ),
                }
            ), 404

        blob_client = (
            container_client
            .get_blob_client(
                blob_name
            )
        )

        blob_download = (
            blob_client.download_blob()
        )

        photo_bytes = (
            blob_download.readall()
        )

        blob_properties = (
            blob_client.get_blob_properties()
        )

        content_type = (
            blob_properties
            .content_settings
            .content_type
            or "image/jpeg"
        )

        response = Response(
            photo_bytes,
            status=200,
            mimetype=content_type,
        )

        response.headers[
            "Content-Disposition"
        ] = (
            "inline; "
            f'filename="attendance_'
            f'{attendance_id}.jpg"'
        )

        response.headers[
            "Cache-Control"
        ] = (
            "private, no-store, "
            "max-age=0"
        )

        return response

    except ResourceNotFoundError:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Attendance photo was not "
                    "found in storage"
                ),
            }
        ), 404

    except SQLAlchemyError as error:
        print(
            (
                "Attendance photo database "
                f"error: {error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not load attendance "
                    "photo information"
                ),
            }
        ), 500

    except Exception as error:
        print(
            (
                "Attendance photo error: "
                f"{repr(error)}"
            ),
            flush=True,
        )

        traceback.print_exc()

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not load attendance "
                    "photo"
                ),
            }
        ), 500
# ---------------------------------------------------------
# Attendance
# ---------------------------------------------------------
@app.route(
    "/attendance",
    methods=["GET"],
)
def get_attendance():
    selected_date_text = (
        request.args.get("date")
    )

    class_id = request.args.get(
        "class_id"
    )

    if (
        not selected_date_text
        or not class_id
    ):
        return jsonify(
            {
                "success": True,
                "attendance": [],
                "message": (
                    "Please select a class "
                    "and date"
                ),
            }
        ), 200

    selected_date = parse_date(
        selected_date_text
    )

    if not selected_date:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Invalid date. "
                    "Use YYYY-MM-DD."
                ),
            }
        ), 400

    try:
        class_id = int(class_id)

    except ValueError:
        return jsonify(
            {
                "success": False,
                "error": "Invalid class ID",
            }
        ), 400

    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        attendance.id,
                        attendance.student_id,
                        attendance.student_name,
                        attendance.timestamp,
                        attendance.status,
                        attendance.attendance_photo_path,
                        attendance.recognition_distance,
                        classes.course_code,
                        classes.course_name
                    FROM attendance
                    LEFT JOIN classes
                        ON attendance.class_id =
                           classes.id
                    WHERE CAST(
                        attendance.timestamp
                        AS DATE
                    ) = :selected_date
                      AND attendance.class_id =
                          :class_id
                    ORDER BY
                        attendance.timestamp DESC
                    """
                ),
                {
                    "selected_date": (
                        selected_date
                    ),
                    "class_id": class_id,
                },
            ).mappings().all()

        attendance_records = []

        for row in rows:
            serialized = serialize_row(
                row
            )

            has_photo = bool(
                row[
                    "attendance_photo_path"
                ]
            )

            serialized[
                "attendance_photo_available"
            ] = has_photo

            serialized[
                "attendance_photo_url"
            ] = (
                f"/attendance/"
                f"{row['id']}/photo"
                if has_photo
                else None
            )

            attendance_records.append(
                serialized
            )

        return jsonify(
            {
                "success": True,
                "attendance": (
                    attendance_records
                ),
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            (
                "Get attendance error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not load attendance"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Dashboard
# ---------------------------------------------------------
@app.route(
    "/dashboard-stats",
    methods=["GET"],
)
def dashboard_stats():
    selected_date_text = (
        request.args.get("date")
    )

    if selected_date_text:
        selected_date = parse_date(
            selected_date_text
        )

        if not selected_date:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Invalid date. "
                        "Use YYYY-MM-DD."
                    ),
                }
            ), 400

    else:
        selected_date = datetime.now(
            timezone.utc
        ).date()

    try:
        with engine.connect() as connection:
            total_students = (
                connection.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM students
                        """
                    )
                ).scalar_one()
            )

            total_attendance_records = (
                connection.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM attendance
                        """
                    )
                ).scalar_one()
            )

            selected_date_attendance = (
                connection.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM attendance
                        WHERE CAST(
                            timestamp AS DATE
                        ) = :selected_date
                        """
                    ),
                    {
                        "selected_date": (
                            selected_date
                        ),
                    },
                ).scalar_one()
            )

            latest = connection.execute(
                text(
                    """
                    SELECT
                        id,
                        student_name,
                        student_id,
                        timestamp,
                        status,
                        attendance_photo_path,
                        recognition_distance
                    FROM attendance
                    WHERE CAST(
                        timestamp AS DATE
                    ) = :selected_date
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """
                ),
                {
                    "selected_date": (
                        selected_date
                    ),
                },
            ).mappings().first()

        latest_attendance = None

        if latest:
            latest_attendance = (
                serialize_row(latest)
            )

            has_photo = bool(
                latest[
                    "attendance_photo_path"
                ]
            )

            latest_attendance[
                "attendance_photo_available"
            ] = has_photo

            latest_attendance[
                "attendance_photo_url"
            ] = (
                f"/attendance/"
                f"{latest['id']}/photo"
                if has_photo
                else None
            )

        return jsonify(
            {
                "success": True,
                "selected_date": (
                    selected_date.isoformat()
                ),
                "total_students": (
                    total_students
                ),
                "total_attendance_records": (
                    total_attendance_records
                ),
                "selected_date_attendance": (
                    selected_date_attendance
                ),
                "latest_attendance": (
                    latest_attendance
                ),
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            (
                "Dashboard error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not load dashboard "
                    "statistics"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Attendance policies
# ---------------------------------------------------------
@app.route(
    "/attendance-policies",
    methods=["POST"],
)
def create_attendance_policy():
    data = (
        request.get_json(silent=True)
        or request.form
    )

    class_id = data.get(
        "class_id"
    )

    policy_name = data.get(
        "policy_name"
    )

    if not class_id or not policy_name:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Class and policy name "
                    "are required"
                ),
            }
        ), 400

    try:
        values = {
            "class_id": int(class_id),

            "policy_name": (
                policy_name
            ),

            "absence_limit": optional_int(
                data.get("absence_limit")
            ),

            "late_limit": optional_int(
                data.get("late_limit")
            ),

            "late_minutes": optional_int(
                data.get("late_minutes")
            ),

            "attendance_weight": optional_float(
                data.get(
                    "attendance_weight"
                )
            ),

            "consequence": data.get(
                "consequence"
            ),

            "excuse_counts": (
                data.get("excuse_counts")
                or "No"
            ),
        }

    except ValueError:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Policy numeric fields "
                    "contain invalid values"
                ),
            }
        ), 400

    try:
        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO attendance_policies
                        (
                            class_id,
                            policy_name,
                            absence_limit,
                            late_limit,
                            late_minutes,
                            attendance_weight,
                            consequence,
                            excuse_counts
                        )
                    VALUES
                        (
                            :class_id,
                            :policy_name,
                            :absence_limit,
                            :late_limit,
                            :late_minutes,
                            :attendance_weight,
                            :consequence,
                            :excuse_counts
                        )
                    """
                ),
                values,
            )

        return jsonify(
            {
                "success": True,
            }
        ), 201

    except SQLAlchemyError as error:
        print(
            (
                "Create policy error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not create policy"
                ),
            }
        ), 500


@app.route(
    (
        "/attendance-policies/"
        "<int:class_id>"
    ),
    methods=["GET"],
)
def get_attendance_policies(class_id):
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        id,
                        class_id,
                        policy_name,
                        absence_limit,
                        late_limit,
                        late_minutes,
                        attendance_weight,
                        consequence,
                        excuse_counts
                    FROM attendance_policies
                    WHERE class_id = :class_id
                    ORDER BY id DESC
                    """
                ),
                {
                    "class_id": class_id,
                },
            ).mappings().all()

        return jsonify(
            {
                "success": True,
                "policies": [
                    serialize_row(row)
                    for row in rows
                ],
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            f"Get policies error: {error}",
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not load policies"
                ),
            }
        ), 500


@app.route(
    (
        "/attendance-policies/"
        "<int:policy_id>"
    ),
    methods=["PUT"],
)
def update_attendance_policy(policy_id):
    data = (
        request.get_json(silent=True)
        or {}
    )

    if not data.get("policy_name"):
        return jsonify(
            {
                "success": False,
                "error": (
                    "Policy name is required"
                ),
            }
        ), 400

    try:
        values = {
            "policy_name": data.get(
                "policy_name"
            ),

            "absence_limit": optional_int(
                data.get("absence_limit")
            ),

            "late_limit": optional_int(
                data.get("late_limit")
            ),

            "late_minutes": optional_int(
                data.get("late_minutes")
            ),

            "attendance_weight": optional_float(
                data.get(
                    "attendance_weight"
                )
            ),

            "consequence": data.get(
                "consequence"
            ),

            "excuse_counts": (
                data.get("excuse_counts")
                or "No"
            ),

            "policy_id": policy_id,
        }

    except ValueError:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Policy numeric fields "
                    "contain invalid values"
                ),
            }
        ), 400

    try:
        with engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    UPDATE attendance_policies
                    SET
                        policy_name =
                            :policy_name,
                        absence_limit =
                            :absence_limit,
                        late_limit =
                            :late_limit,
                        late_minutes =
                            :late_minutes,
                        attendance_weight =
                            :attendance_weight,
                        consequence =
                            :consequence,
                        excuse_counts =
                            :excuse_counts
                    WHERE id = :policy_id
                    """
                ),
                values,
            )

        return jsonify(
            {
                "success": True,
                "updated": (
                    result.rowcount > 0
                ),
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            (
                "Update policy error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not update policy"
                ),
            }
        ), 500


@app.route(
    (
        "/attendance-policies/"
        "<int:policy_id>"
    ),
    methods=["DELETE"],
)
def delete_attendance_policy(policy_id):
    try:
        with engine.begin() as connection:
            result = connection.execute(
                text(
                    """
                    DELETE FROM
                        attendance_policies
                    WHERE id = :policy_id
                    """
                ),
                {
                    "policy_id": policy_id,
                },
            )

        return jsonify(
            {
                "success": True,
                "deleted": (
                    result.rowcount > 0
                ),
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            (
                "Delete policy error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not delete policy"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Policy risk
# ---------------------------------------------------------
@app.route(
    "/policy-risk/<int:class_id>",
    methods=["GET"],
)
def policy_risk(class_id):
    try:
        with engine.connect() as connection:
            policy = connection.execute(
                text(
                    """
                    SELECT
                        id,
                        policy_name,
                        absence_limit,
                        consequence
                    FROM attendance_policies
                    WHERE class_id = :class_id
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ),
                {
                    "class_id": class_id,
                },
            ).mappings().first()

            if not policy:
                return jsonify(
                    {
                        "success": True,
                        "has_policy": False,
                        "message": (
                            "No attendance policy "
                            "set for this class"
                        ),
                        "at_risk": [],
                    }
                ), 200

            enrolled_students = (
                connection.execute(
                    text(
                        """
                        SELECT
                            students.student_id,
                            students.student_name
                        FROM class_students
                        JOIN students
                            ON class_students.student_id =
                               students.student_id
                        WHERE class_students.class_id =
                              :class_id
                        """
                    ),
                    {
                        "class_id": class_id,
                    },
                ).mappings().all()
            )

            total_class_days = (
                connection.execute(
                    text(
                        """
                        SELECT COUNT(
                            DISTINCT CAST(
                                timestamp AS DATE
                            )
                        )
                        FROM attendance
                        WHERE class_id =
                              :class_id
                        """
                    ),
                    {
                        "class_id": class_id,
                    },
                ).scalar_one()
            )

            at_risk = []

            for student in enrolled_students:
                present_days = (
                    connection.execute(
                        text(
                            """
                            SELECT COUNT(
                                DISTINCT CAST(
                                    timestamp AS DATE
                                )
                            )
                            FROM attendance
                            WHERE class_id =
                                  :class_id
                              AND student_id =
                                  :student_id
                              AND status =
                                  'Present'
                            """
                        ),
                        {
                            "class_id": (
                                class_id
                            ),
                            "student_id": (
                                student[
                                    "student_id"
                                ]
                            ),
                        },
                    ).scalar_one()
                )

                absences = max(
                    total_class_days
                    - present_days,
                    0,
                )

                absence_limit = (
                    policy[
                        "absence_limit"
                    ]
                )

                if (
                    absence_limit is not None
                    and absences
                    >= absence_limit
                ):
                    at_risk.append(
                        {
                            "student_id": (
                                student[
                                    "student_id"
                                ]
                            ),
                            "student_name": (
                                student[
                                    "student_name"
                                ]
                            ),
                            "absences": (
                                absences
                            ),
                            "absence_limit": (
                                absence_limit
                            ),
                        }
                    )

        return jsonify(
            {
                "success": True,
                "has_policy": True,
                "policy_id": policy["id"],
                "policy_name": (
                    policy["policy_name"]
                ),
                "absence_limit": (
                    policy["absence_limit"]
                ),
                "consequence": (
                    policy["consequence"]
                ),
                "total_class_days": (
                    total_class_days
                ),
                "at_risk": at_risk,
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            (
                "Policy risk error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not calculate "
                    "policy risk"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Attendance trend
# ---------------------------------------------------------
@app.route(
    (
        "/attendance-trend/"
        "<int:class_id>"
    ),
    methods=["GET"],
)
def attendance_trend(class_id):
    try:
        with engine.connect() as connection:
            total_students = (
                connection.execute(
                    text(
                        """
                        SELECT COUNT(*)
                        FROM class_students
                        WHERE class_id =
                              :class_id
                        """
                    ),
                    {
                        "class_id": class_id,
                    },
                ).scalar_one()
            )

            rows = connection.execute(
                text(
                    """
                    SELECT
                        CAST(timestamp AS DATE)
                            AS attendance_date,
                        COUNT(
                            DISTINCT student_id
                        )
                            AS present_count
                    FROM attendance
                    WHERE class_id = :class_id
                      AND status = 'Present'
                    GROUP BY CAST(
                        timestamp AS DATE
                    )
                    ORDER BY CAST(
                        timestamp AS DATE
                    ) DESC
                    LIMIT 5
                    """
                ),
                {
                    "class_id": class_id,
                },
            ).mappings().all()

        trend = []

        for row in reversed(rows):
            present_count = (
                row["present_count"]
            )

            attendance_rate = 0

            if total_students > 0:
                attendance_rate = round(
                    (
                        present_count
                        / total_students
                    )
                    * 100
                )

            trend.append(
                {
                    "date": (
                        row[
                            "attendance_date"
                        ].isoformat()
                    ),
                    "present_count": (
                        present_count
                    ),
                    "total_students": (
                        total_students
                    ),
                    "attendance_rate": (
                        attendance_rate
                    ),
                }
            )

        return jsonify(
            {
                "success": True,
                "trend": trend,
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            (
                "Attendance trend error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not load "
                    "attendance trend"
                ),
            }
        ), 500


# ---------------------------------------------------------
# CSV export
# ---------------------------------------------------------
@app.route(
    "/export-attendance",
    methods=["GET"],
)
def export_attendance():
    class_id = request.args.get(
        "class_id"
    )

    selected_date_text = (
        request.args.get("date")
    )

    if not class_id or not selected_date_text:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Class and date are required"
                ),
            }
        ), 400

    selected_date = parse_date(
        selected_date_text
    )

    if not selected_date:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Invalid date. "
                    "Use YYYY-MM-DD."
                ),
            }
        ), 400

    try:
        class_id = int(class_id)

    except ValueError:
        return jsonify(
            {
                "success": False,
                "error": "Invalid class ID",
            }
        ), 400

    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        classes.course_code,
                        classes.course_name,
                        attendance.student_id,
                        attendance.student_name,
                        attendance.timestamp,
                        attendance.status,
                        attendance.recognition_distance,
                        attendance.id
                    FROM attendance
                    JOIN classes
                        ON attendance.class_id =
                           classes.id
                    WHERE attendance.class_id =
                          :class_id
                      AND CAST(
                          attendance.timestamp
                          AS DATE
                      ) = :selected_date
                    ORDER BY
                        attendance.timestamp ASC
                    """
                ),
                {
                    "class_id": class_id,
                    "selected_date": (
                        selected_date
                    ),
                },
            ).all()

    except SQLAlchemyError as error:
        print(
            (
                "Export attendance error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not export attendance"
                ),
            }
        ), 500

    output = io.StringIO()

    writer = csv.writer(
        output
    )

    writer.writerow(
        [
            "Course Code",
            "Course Name",
            "Student ID",
            "Student Name",
            "Timestamp",
            "Status",
            "Recognition Distance",
            "Attendance Photo URL",
        ]
    )

    for row in rows:
        photo_url = (
            f"/attendance/{row[7]}/photo"
        )

        writer.writerow(
            [
                serialize_value(row[0]),
                serialize_value(row[1]),
                serialize_value(row[2]),
                serialize_value(row[3]),
                serialize_value(row[4]),
                serialize_value(row[5]),
                serialize_value(row[6]),
                photo_url,
            ]
        )

    response = app.response_class(
        response=output.getvalue(),
        status=200,
        mimetype="text/csv",
    )

    response.headers[
        "Content-Disposition"
    ] = (
        "attachment; "
        f"filename=attendance_"
        f"{class_id}_"
        f"{selected_date.isoformat()}"
        ".csv"
    )

    return response


# ---------------------------------------------------------
# Excuses
# ---------------------------------------------------------
@app.route(
    "/excuses",
    methods=["POST"],
)
def submit_excuse():
    data = (
        request.get_json(silent=True)
        or request.form
    )

    student_id = data.get(
        "student_id"
    )

    class_id = data.get(
        "class_id"
    )

    excuse_date_text = data.get(
        "excuse_date"
    )

    reason = data.get(
        "reason"
    )

    if (
        not student_id
        or not class_id
        or not excuse_date_text
        or not reason
    ):
        return jsonify(
            {
                "success": False,
                "error": (
                    "Missing required fields"
                ),
            }
        ), 400

    excuse_date = parse_date(
        excuse_date_text
    )

    if not excuse_date:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Invalid date. "
                    "Use YYYY-MM-DD."
                ),
            }
        ), 400

    try:
        class_id = int(class_id)

    except ValueError:
        return jsonify(
            {
                "success": False,
                "error": "Invalid class ID",
            }
        ), 400

    try:
        with engine.begin() as connection:
            student_exists = (
                connection.execute(
                    text(
                        """
                        SELECT student_id
                        FROM students
                        WHERE student_id =
                              :student_id
                        """
                    ),
                    {
                        "student_id": (
                            student_id
                        ),
                    },
                ).first()
            )

            if not student_exists:
                return jsonify(
                    {
                        "success": False,
                        "error": (
                            "Student not found"
                        ),
                    }
                ), 404

            class_exists = (
                connection.execute(
                    text(
                        """
                        SELECT id
                        FROM classes
                        WHERE id = :class_id
                        """
                    ),
                    {
                        "class_id": (
                            class_id
                        ),
                    },
                ).first()
            )

            if not class_exists:
                return jsonify(
                    {
                        "success": False,
                        "error": (
                            "Class not found"
                        ),
                    }
                ), 404

            connection.execute(
                text(
                    """
                    INSERT INTO excuses
                        (
                            student_id,
                            class_id,
                            excuse_date,
                            reason,
                            status
                        )
                    VALUES
                        (
                            :student_id,
                            :class_id,
                            :excuse_date,
                            :reason,
                            :status
                        )
                    """
                ),
                {
                    "student_id": student_id,
                    "class_id": class_id,
                    "excuse_date": excuse_date,
                    "reason": reason,
                    "status": "Pending",
                },
            )

        return jsonify(
            {
                "success": True,
            }
        ), 201

    except SQLAlchemyError as error:
        print(
            (
                "Submit excuse error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not submit excuse"
                ),
            }
        ), 500


@app.route(
    "/excuses",
    methods=["GET"],
)
def get_excuses():
    try:
        with engine.connect() as connection:
            rows = connection.execute(
                text(
                    """
                    SELECT
                        excuses.id,
                        excuses.student_id,
                        students.student_name,
                        excuses.class_id,
                        classes.course_code,
                        classes.course_name,
                        excuses.excuse_date,
                        excuses.reason,
                        excuses.status,
                        excuses.submitted_at
                    FROM excuses
                    JOIN students
                        ON excuses.student_id =
                           students.student_id
                    JOIN classes
                        ON excuses.class_id =
                           classes.id
                    ORDER BY
                        excuses.submitted_at DESC
                    """
                )
            ).mappings().all()

        return jsonify(
            {
                "success": True,
                "excuses": [
                    serialize_row(row)
                    for row in rows
                ],
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            f"Get excuses error: {error}",
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not load excuses"
                ),
            }
        ), 500


@app.route(
    "/excuses/<int:excuse_id>",
    methods=["PUT"],
)
def update_excuse_status(excuse_id):
    data = (
        request.get_json(silent=True)
        or {}
    )

    status = data.get(
        "status"
    )

    if status not in [
        "Approved",
        "Rejected",
        "Pending",
    ]:
        return jsonify(
            {
                "success": False,
                "error": "Invalid status",
            }
        ), 400

    try:
        with engine.begin() as connection:
            excuse = connection.execute(
                text(
                    """
                    SELECT
                        student_id,
                        class_id,
                        excuse_date
                    FROM excuses
                    WHERE id = :excuse_id
                    """
                ),
                {
                    "excuse_id": (
                        excuse_id
                    ),
                },
            ).mappings().first()

            if not excuse:
                return jsonify(
                    {
                        "success": False,
                        "error": (
                            "Excuse not found"
                        ),
                    }
                ), 404

            connection.execute(
                text(
                    """
                    UPDATE excuses
                    SET status = :status
                    WHERE id = :excuse_id
                    """
                ),
                {
                    "status": status,
                    "excuse_id": excuse_id,
                },
            )

            if status == "Approved":
                student_name = (
                    connection.execute(
                        text(
                            """
                            SELECT student_name
                            FROM students
                            WHERE student_id =
                                  :student_id
                            """
                        ),
                        {
                            "student_id": (
                                excuse[
                                    "student_id"
                                ]
                            ),
                        },
                    ).scalar_one_or_none()
                    or "Unknown"
                )

                attendance_record = (
                    connection.execute(
                        text(
                            """
                            SELECT id
                            FROM attendance
                            WHERE student_id =
                                  :student_id
                              AND class_id =
                                  :class_id
                              AND CAST(
                                  timestamp AS DATE
                              ) = :excuse_date
                            LIMIT 1
                            """
                        ),
                        {
                            "student_id": (
                                excuse[
                                    "student_id"
                                ]
                            ),
                            "class_id": (
                                excuse[
                                    "class_id"
                                ]
                            ),
                            "excuse_date": (
                                excuse[
                                    "excuse_date"
                                ]
                            ),
                        },
                    ).first()
                )

                if attendance_record:
                    connection.execute(
                        text(
                            """
                            UPDATE attendance
                            SET status = 'Excused'
                            WHERE id =
                                  :attendance_id
                            """
                        ),
                        {
                            "attendance_id": (
                                attendance_record[0]
                            ),
                        },
                    )

                else:
                    excuse_timestamp = (
                        datetime.combine(
                            excuse[
                                "excuse_date"
                            ],
                            time.min,
                            tzinfo=timezone.utc,
                        )
                    )

                    connection.execute(
                        text(
                            """
                            INSERT INTO attendance
                                (
                                    class_id,
                                    student_id,
                                    student_name,
                                    timestamp,
                                    status,
                                    attendance_photo_path,
                                    recognition_distance
                                )
                            VALUES
                                (
                                    :class_id,
                                    :student_id,
                                    :student_name,
                                    :timestamp,
                                    :status,
                                    NULL,
                                    NULL
                                )
                            """
                        ),
                        {
                            "class_id": (
                                excuse[
                                    "class_id"
                                ]
                            ),
                            "student_id": (
                                excuse[
                                    "student_id"
                                ]
                            ),
                            "student_name": (
                                student_name
                            ),
                            "timestamp": (
                                excuse_timestamp
                            ),
                            "status": "Excused",
                        },
                    )

        return jsonify(
            {
                "success": True,
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            (
                "Update excuse error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not update excuse"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Frontend
# ---------------------------------------------------------
@app.route("/")
def serve_home():
    return send_from_directory(
        FRONTEND_DIR,
        "index.html",
    )


@app.route(
    "/<path:filename>"
)
def serve_frontend(filename):
    return send_from_directory(
        FRONTEND_DIR,
        filename,
    )


# ---------------------------------------------------------
# Local development only
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",

        port=int(
            os.environ.get(
                "PORT",
                5000,
            )
        ),

        debug=(
            os.environ.get(
                "FLASK_DEBUG",
                "false",
            ).lower()
            == "true"
        ),
    )