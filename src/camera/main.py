import cv2
import face_recognition
import numpy as np
import psycopg2
import os
import uuid

# Conectar a la base de datos
conn = psycopg2.connect("dbname=cerezavision user=postgres password=takane")
cur = conn.cursor()

# Crear la tabla si no existe
cur.execute('''CREATE TABLE IF NOT EXISTS faces (id text PRIMARY KEY, name TEXT, auth INT DEFAULT 0, encoding BYTEA);''')

# Inicializar la cámara
cap = cv2.VideoCapture(0)

# Crear un directorio para guardar las capturas de rostros desconocidos si no existe
if not os.path.exists('unknown_faces'):
    os.makedirs('unknown_faces')

while True:
    # Leer el cuadro de la cámara
    ret, frame = cap.read()

    # Convertir la imagen a RGB
    rgb_frame = frame[:, :, ::-1]

    # Detectar rostros en la imagen
    face_locations = face_recognition.face_locations(rgb_frame)

    # Verificar que se hayan detectado rostros
    if len(face_locations) > 0:
        # Convertir la imagen a formato adecuado
        rgb_frame = np.array(rgb_frame, dtype=np.uint8)

        # Convertir las coordenadas de los rostros a formato adecuado
        face_locations = [(top, right, bottom, left)
                          for (top, right, bottom, left) in face_locations]

        # Calcular las codificaciones de los rostros
        face_encodings = face_recognition.face_encodings(
            rgb_frame, face_locations)
    else:
        # No se detectaron rostros, manejar el caso según sea necesario
        face_encodings = []

    # Comparar las codificaciones de rostros detectadas con las codificaciones conocidas
    for i, face_encoding in enumerate(face_encodings):
        # Cargar las codificaciones de rostros y los nombres correspondientes desde la base de datos
        known_face_encodings = []
        known_face_names = {}
        cur.execute('SELECT id, name, encoding FROM faces')
        rows = cur.fetchall()
        for row in rows:
            id, name, encoding = row
            known_face_names[id] = name
            known_face_encodings.append(
                (id, np.frombuffer(encoding, dtype=np.float64)))

        matches = face_recognition.compare_faces(
            [encoding for id, encoding in known_face_encodings], face_encoding)
        name = "Desconocido"
        id = None

        # Buscar el nombre correspondiente a la codificación de rostro coincidente
        if True in matches:
            first_match_index = matches.index(True)
            id, encoding = known_face_encodings[first_match_index]
            name = known_face_names[id]
        else:
            # Generar un ID único para la nueva cara y guardarla en la base de datos
            id = str(uuid.uuid4())
            cur.execute("INSERT INTO faces (id, name, encoding) VALUES (%s, %s, %s)",
                        (id, name, face_encoding.tobytes()))
            conn.commit()
            known_face_names[id] = name
            known_face_encodings.append((id, face_encoding))

            # Guardar una captura del rostro desconocido
            top, right, bottom, left = face_locations[i]
            face_image = frame[top:bottom, left:right]
            cv2.imwrite(f'unknown_faces/{id}.png', face_image)

        # Dibujar rectángulos y mostrar el nombre de la persona
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Mostrar la imagen resultante
    cv2.imshow('Face Recognition', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar las ventanas
cap.release()
cv2.destroyAllWindows()
cur.close()
conn.close()
