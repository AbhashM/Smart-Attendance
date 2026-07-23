from flask import (
    Flask,
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
import cv2
import numpy as np
import traceback
import io
import os
import tempfile
import threading
import uuid

from azure.core.exceptions import ResourceExistsError
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


app = Flask(__name__)
CORS(app)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
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
            "Missing database environment variables: "
            + ", ".join(missing_variables)
        )

    return URL.create(
        drivername="postgresql+psycopg",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.environ["DB_HOST"],
        port=int(
            os.environ.get(
                "DB_PORT",
                "5432",
            )
        ),
        database=os.environ["DB_NAME"],
        query={
            "sslmode": os.environ.get(
                "DB_SSLMODE",
                "require",
            )
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
        "AZURE_STORAGE_CONNECTION_STRING "
        "is missing"
    )


blob_service_client = (
    BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )
)

container_client = (
    blob_service_client.get_container_client(
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
        DateTime(timezone=True),
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
        DateTime(timezone=True),
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
    ),

    Column(
        "late_limit",
        Integer,
    ),

    Column(
        "late_minutes",
        Integer,
    ),

    Column(
        "attendance_weight",
        Float,
    ),

    Column(
        "consequence",
        Text,
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
        DateTime(timezone=True),
        nullable=False,

        server_default=text(
            "CURRENT_TIMESTAMP"
        ),
    ),
)


metadata.create_all(engine)


# ---------------------------------------------------------
# Lazy-loaded AI dependencies
# ---------------------------------------------------------
_deepface = None
_face_detector = None
_ai_lock = threading.Lock()


def get_deepface():
    global _deepface

    if _deepface is None:
        with _ai_lock:
            if _deepface is None:
                print(
                    "Loading DeepFace...",
                    flush=True,
                )

                from deepface import DeepFace

                _deepface = DeepFace

                print(
                    "DeepFace loaded.",
                    flush=True,
                )

    return _deepface


def get_face_detector():
    global _face_detector

    if _face_detector is None:
        with _ai_lock:
            if _face_detector is None:
                print(
                    "Loading MediaPipe "
                    "face detector...",
                    flush=True,
                )

                import mediapipe as mp

                _face_detector = (
                    mp.solutions
                    .face_detection
                    .FaceDetection(
                        model_selection=0,
                        min_detection_confidence=0.5,
                    )
                )

                print(
                    "MediaPipe face detector "
                    "loaded.",
                    flush=True,
                )

    return _face_detector


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def safe_remove(path):
    try:
        if path and os.path.exists(path):
            os.remove(path)

    except OSError as error:
        print(
            f"Could not remove {path}: {error}",
            flush=True,
        )


def parse_date(value):
    try:
        return date.fromisoformat(value)

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
            date,
            datetime,
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
    return "".join(
        character
        if (
            character.isalnum()
            or character in "-_"
        )
        else "_"
        for character in value
    )


def upload_face_image(
    image_file,
    student_id,
    appearance_label,
):
    original_filename = os.path.basename(
        image_file.filename
        or "image.jpg"
    )

    extension = os.path.splitext(
        original_filename
    )[1].lower()

    if not extension:
        extension = ".jpg"

    safe_label = sanitize_label(
        appearance_label
    )

    blob_name = (
        f"{student_id}/"
        f"{safe_label}/"
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    image_file.stream.seek(0)

    blob_client = (
        container_client.get_blob_client(
            blob_name
        )
    )

    blob_client.upload_blob(
        image_file.stream,
        overwrite=True,

        content_settings=ContentSettings(
            content_type=(
                image_file.content_type
                or "image/jpeg"
            )
        ),
    )

    return blob_name


def download_blob_to_temp(blob_name):
    extension = os.path.splitext(
        blob_name
    )[1]

    temporary_file = (
        tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension or ".jpg",
        )
    )

    temporary_path = temporary_file.name

    try:
        blob_client = (
            container_client.get_blob_client(
                blob_name
            )
        )

        blob_data = (
            blob_client.download_blob()
        )

        temporary_file.write(
            blob_data.readall()
        )

        temporary_file.close()

        return temporary_path

    except Exception:
        temporary_file.close()

        safe_remove(
            temporary_path
        )

        raise


def delete_blob_if_exists(blob_name):
    if not blob_name:
        return

    try:
        container_client.delete_blob(
            blob_name
        )

    except Exception as error:
        print(
            (
                "Could not delete blob "
                f"{blob_name}: {error}"
            ),
            flush=True,
        )
def download_blob_as_image(blob_name):
    blob_client = container_client.get_blob_client(
        blob_name
    )

    blob_bytes = blob_client.download_blob().readall()

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
            f"Could not decode blob image: {blob_name}"
        )

    return image

# ---------------------------------------------------------
# API and health
# ---------------------------------------------------------
@app.route(
    "/api",
    methods=["GET"],
)
def home():
    return jsonify(
        {
            "status": "running",
            "service": (
                "Smart Attendance Backend"
            ),
            "database": "PostgreSQL",
            "image_storage": (
                "Azure Blob Storage"
            ),
        }
    ), 200


@app.route(
    "/health",
    methods=["GET"],
)
def health():
    try:
        with engine.connect() as connection:
            connection.execute(
                text("SELECT 1")
            )

        container_client.get_container_properties()

        return jsonify(
            {
                "success": True,
                "status": "healthy",
                "database": "connected",
                "blob_storage": "connected",
            }
        ), 200

    except Exception as error:
        print(
            f"Health check error: {error}",
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "status": "unhealthy",
                "error": str(error),
            }
        ), 503


# ---------------------------------------------------------
# Face detection
# ---------------------------------------------------------
@app.route(
    "/detect",
    methods=["POST"],
)
def detect_face():
    if "image" not in request.files:
        return jsonify(
            {
                "success": False,
                "error": "No image uploaded",
            }
        ), 400

    try:
        import cv2
        import numpy as np

        uploaded_file = (
            request.files["image"]
        )

        image_bytes = (
            uploaded_file.read()
        )

        numpy_image = np.frombuffer(
            image_bytes,
            np.uint8,
        )

        image = cv2.imdecode(
            numpy_image,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Invalid image file"
                    ),
                }
            ), 400

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        detector = get_face_detector()

        results = detector.process(
            rgb_image
        )

        return jsonify(
            {
                "success": True,
                "face_detected": bool(
                    results.detections
                ),
            }
        ), 200

    except Exception as error:
        print(
            (
                "Face detection error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Face detection failed"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Register student
# ---------------------------------------------------------
@app.route(
    "/register",
    methods=["POST"],
)
def register_student():
    student_name = request.form.get(
        "student_name"
    )

    student_id = request.form.get(
        "student_id"
    )

    image = request.files.get(
        "image"
    )

    if (
        not student_name
        or not student_id
        or not image
    ):
        return jsonify(
            {
                "success": False,
                "error": (
                    "Missing required fields"
                ),
            }
        ), 400

    blob_name = None

    try:
        blob_name = upload_face_image(
            image,
            student_id,
            "Default",
        )

        with engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO students
                        (
                            student_name,
                            student_id,
                            image_path
                        )
                    VALUES
                        (
                            :student_name,
                            :student_id,
                            :image_path
                        )
                    """
                ),
                {
                    "student_name": (
                        student_name
                    ),
                    "student_id": student_id,
                    "image_path": blob_name,
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
                    "image_path": blob_name,
                    "appearance_label": (
                        "Default"
                    ),
                },
            )

        return jsonify(
            {
                "success": True,
                "image_blob": blob_name,
            }
        ), 201

    except IntegrityError:
        delete_blob_if_exists(
            blob_name
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Student ID already exists"
                ),
            }
        ), 400

    except Exception as error:
        delete_blob_if_exists(
            blob_name
        )

        print(
            f"Registration error: {error}",
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not register student"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Alternate appearances
# ---------------------------------------------------------
@app.route(
    "/add-appearance",
    methods=["POST"],
)
def add_appearance():
    student_id = request.form.get(
        "student_id"
    )

    appearance_label = (
        request.form.get(
            "appearance_label"
        )
        or "Alternate"
    )

    image = request.files.get(
        "image"
    )

    if not student_id or not image:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Missing student ID or image"
                ),
            }
        ), 400

    blob_name = None

    try:
        with engine.connect() as connection:
            student = connection.execute(
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

        if not student:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "Student not found"
                    ),
                }
            ), 404

        blob_name = upload_face_image(
            image,
            student_id,
            appearance_label,
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
                    "image_path": blob_name,
                    "appearance_label": (
                        appearance_label
                    ),
                },
            )

        return jsonify(
            {
                "success": True,
                "image_blob": blob_name,
            }
        ), 201

    except Exception as error:
        delete_blob_if_exists(
            blob_name
        )

        print(
            (
                "Add appearance error: "
                f"{error}"
            ),
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not add appearance"
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
            rows = connection.execute(
                text(
                    """
                    SELECT
                        student_id,
                        student_name
                    FROM students
                    ORDER BY student_name ASC
                    """
                )
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
            f"Get students error: {error}",
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

    course_name = data.get(
        "course_name"
    )

    course_code = data.get(
        "course_code"
    )

    professor_name = data.get(
        "professor_name"
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
                    "Missing required fields"
                ),
            }
        ), 400

    try:
        with engine.begin() as connection:
            class_id = connection.execute(
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
                    "course_name": course_name,
                    "course_code": course_code,
                    "professor_name": (
                        professor_name
                    ),
                },
            ).scalar_one()

        return jsonify(
            {
                "success": True,
                "class_id": class_id,
            }
        ), 201

    except IntegrityError:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Course code already exists"
                ),
            }
        ), 400

    except SQLAlchemyError as error:
        print(
            f"Create class error: {error}",
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
                        id,
                        course_name,
                        course_code,
                        professor_name
                    FROM classes
                    ORDER BY id DESC
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
            f"Get classes error: {error}",
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
# Class students
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

    student_id = data.get(
        "student_id"
    )

    if not class_id or not student_id:
        return jsonify(
            {
                "success": False,
                "error": (
                    "Missing class ID or student ID"
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
            class_exists = connection.execute(
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
        ), 400

    except SQLAlchemyError as error:
        print(
            f"Enrollment error: {error}",
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not enroll student"
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
                        students.student_name
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
                    "Could not load class students"
                ),
            }
        ), 500


@app.route(
    (
        "/class-students/"
        "<int:class_id>/<student_id>"
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

        return jsonify(
            {
                "success": True,
                "removed": (
                    result.rowcount > 0
                ),
            }
        ), 200

    except SQLAlchemyError as error:
        print(
            f"Remove student error: {error}",
            flush=True,
        )

        return jsonify(
            {
                "success": False,
                "error": (
                    "Could not remove student"
                ),
            }
        ), 500


# ---------------------------------------------------------
# Face recognition
# ---------------------------------------------------------
@app.route(
    "/recognize",
    methods=["POST"],
)
def recognize_student():
    if "image" not in request.files:
        return jsonify(
            {
                "success": False,
                "error": "No image uploaded",
            }
        ), 400

    class_id = request.form.get("class_id")

    if not class_id:
        return jsonify(
            {
                "success": False,
                "error": "Please select a class",
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
        # Read the newly uploaded recognition image
        uploaded_image = request.files["image"]
        uploaded_bytes = uploaded_image.read()

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
                    "error": "Uploaded image could not be read",
                }
            ), 400

        print(
            f"Uploaded image shape: {test_image.shape}",
            flush=True,
        )

        # Load students enrolled in the selected class
        with engine.connect() as connection:
            students = connection.execute(
                text(
                    """
                    SELECT
                        students.student_name,
                        students.student_id,
                        student_images.image_path
                    FROM students
                    JOIN student_images
                        ON students.student_id =
                           student_images.student_id
                    JOIN class_students
                        ON students.student_id =
                           class_students.student_id
                    WHERE class_students.class_id =
                          :class_id
                    """
                ),
                {
                    "class_id": class_id,
                },
            ).mappings().all()

        if not students:
            return jsonify(
                {
                    "success": False,
                    "error": (
                        "No student appearance profiles "
                        "found for this class"
                    ),
                }
            ), 400

        deepface = get_deepface()

        # Compare uploaded image with every registered appearance
        for student in students:
            student_name = student["student_name"]
            student_id = student["student_id"]
            blob_name = student["image_path"]

            try:
                print(
                    f"Checking blob: {blob_name}",
                    flush=True,
                )

                stored_image = download_blob_as_image(
                    blob_name
                )

                print(
                    (
                        f"Stored image shape for "
                        f"{student_id}: {stored_image.shape}"
                    ),
                    flush=True,
                )

                result = deepface.verify(
                    img1_path=test_image,
                    img2_path=stored_image,
                    model_name="VGG-Face",
                    detector_backend="skip",
                    enforce_detection=False,
                    silent=True,
                )

                print(
                    (
                        f"DeepFace result for "
                        f"{student_id}: {result}"
                    ),
                    flush=True,
                )

                if not result.get("verified"):
                    continue

                current_time = datetime.now(
                    timezone.utc
                )

                attendance_date = current_time.date()

                with engine.begin() as connection:
                    existing_record = connection.execute(
                        text(
                            """
                            SELECT id
                            FROM attendance
                            WHERE class_id = :class_id
                              AND student_id = :student_id
                              AND CAST(timestamp AS DATE) =
                                  :attendance_date
                            LIMIT 1
                            """
                        ),
                        {
                            "class_id": class_id,
                            "student_id": student_id,
                            "attendance_date": attendance_date,
                        },
                    ).first()

                    if existing_record:
                        connection.execute(
                            text(
                                """
                                UPDATE attendance
                                SET
                                    student_name = :student_name,
                                    timestamp = :timestamp,
                                    status = 'Present'
                                WHERE id = :attendance_id
                                """
                            ),
                            {
                                "student_name": student_name,
                                "timestamp": current_time,
                                "attendance_id": (
                                    existing_record[0]
                                ),
                            },
                        )

                    else:
                        connection.execute(
                            text(
                                """
                                INSERT INTO attendance
                                    (
                                        class_id,
                                        student_id,
                                        student_name,
                                        timestamp,
                                        status
                                    )
                                VALUES
                                    (
                                        :class_id,
                                        :student_id,
                                        :student_name,
                                        :timestamp,
                                        :status
                                    )
                                """
                            ),
                            {
                                "class_id": class_id,
                                "student_id": student_id,
                                "student_name": student_name,
                                "timestamp": current_time,
                                "status": "Present",
                            },
                        )

                return jsonify(
                    {
                        "success": True,
                        "student_name": student_name,
                        "student_id": student_id,
                        "class_id": class_id,
                        "attendance_marked": True,
                        "status": "Present",
                        "timestamp": (
                            current_time.isoformat()
                        ),
                    }
                ), 200

            except Exception as error:
                print(
                    (
                        f"Recognition error for "
                        f"{student_id}: {repr(error)}"
                    ),
                    flush=True,
                )

                traceback.print_exc()

        return jsonify(
            {
                "success": False,
                "error": (
                    "No matching student found "
                    "in selected class"
                ),
            }
        ), 404

    except Exception as error:
        print(
            f"Recognition route error: {repr(error)}",
            flush=True,
        )

        traceback.print_exc()

        return jsonify(
            {
                "success": False,
                "error": "Face recognition failed",
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

        return jsonify(
            {
                "success": True,
                "attendance": [
                    serialize_row(row)
                    for row in rows
                ],
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
                        student_name,
                        student_id,
                        timestamp
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
                    serialize_row(latest)
                    if latest
                    else None
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
                        attendance.status
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
        ]
    )

    for row in rows:
        writer.writerow(
            [
                serialize_value(value)
                for value in row
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
                                    status
                                )
                            VALUES
                                (
                                    :class_id,
                                    :student_id,
                                    :student_name,
                                    :timestamp,
                                    :status
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