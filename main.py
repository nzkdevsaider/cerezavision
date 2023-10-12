import cv2
import face_recognition
import numpy as np
import psycopg2

# Conectar a la base de datos
conn = psycopg2.connect("dbname=cerezavision user=postgres password=takane")
cur = conn.cursor()

# Crear la tabla si no existe
cur.execute('''CREATE TABLE IF NOT EXISTS faces
             (name TEXT, encoding BYTEA)''')

# Cargar las codificaciones de rostros desde la base de datos
known_face_encodings = []
known_face_names = []
cur.execute('SELECT name, encoding FROM faces')
rows = cur.fetchall()
for row in rows:
    name, encoding = row
    known_face_names.append(name)
    known_face_encodings.append(np.frombuffer(encoding, dtype=np.float64))

# Inicializar la cámara
cap = cv2.VideoCapture(0)

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
        face_locations = [(top, right, bottom, left) for (top, right, bottom, left) in face_locations]

        # Calcular las codificaciones de los rostros
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    else:
        # No se detectaron rostros, manejar el caso según sea necesario
        face_encodings = []

    # Comparar las codificaciones de rostros detectadas con las codificaciones conocidas
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Desconocido"

        # Buscar el nombre correspondiente a la codificación de rostro coincidente
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            # Guardar la nueva cara en la base de datos
            name = input("Ingrese el nombre de la persona: ")
            cur.execute("INSERT INTO faces (name, encoding) VALUES (%s, %s)", (name, face_encoding.tobytes()))
            conn.commit()
            known_face_names.append(name)
            known_face_encodings.append(face_encoding)

        # Dibujar rectángulos y mostrar el nombre de la persona
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

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
