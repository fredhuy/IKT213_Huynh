import cv2

# TASK IV
iris_image = cv2.imread("iris-1.jpg", cv2.IMREAD_UNCHANGED )

def print_image_information(image):
    image_shape = image.shape

    print("Image Height:", image_shape[0])
    print("Image Width:", image_shape[1])
    print("Image Channels:", image_shape[2])
    print("Image Size:", image.size)
    print("Image Type:", image.dtype)

def show_image(image):
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#TASK V
def camera_thing():

    # Open the default camera
    cam = cv2.VideoCapture(0)

    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cam.read()

        # Write the frame to the output file
        out.write(frame)

        # Display the captured frame
        cv2.imshow('Camera', frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) == ord('q'):
            break

    #print("CAMERA FPS:", cam.get(cv2.CAP_PROP_FPS))
    #print("CAMERA WIDTH:", frame_width)
    #print("CAMERA HEIGHT:", frame_height)



    with open("camera_outputs.txt", "w") as file:
        file.write(f"CAMERA FPS: {str(cam.get(cv2.CAP_PROP_FPS))} \n")
        file.write(f"CAMERA WIDTH: {str(frame_width)} \n")
        file.write(f"CAMERA HEIGHT: {str(frame_height)}")

    # Release the capture and writer objects
    cam.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    print_image_information(iris_image)
    #show_image(iris_image)

    #camera_thing()


main()
