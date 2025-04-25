import cv2
import numpy as np
import math
# Functions



def angle_between_points(points):
    dx = points[1][0] - points[0][0] 
    dy = points[1][1]  - points[0][1] 
    angle_rad = math.atan2(dy, dx)
    angle_deg = round(math.degrees(angle_rad))*-1
    print(f"angle between point 1 X:{points[0][0]} Y:{points[0][0]} en point 2 X:{points[1][0]} Y:{points[1][0]} is {angle_deg} degrees")
    return angle_deg
def cal_distance(heigt):
    #object_mm_on_sensor = (object_pixels / image_height_pixels) × sensor_height_mm
    #afstand = (echte_grootte_mm × focal lenght) / object_mm_on_sensor
    #deze gegevens zijn voor mijn gsm s23 Ultra nog aan passen voor camera jetank
    object_mm_on_sensor = (heigt / (image.shape[1]/10)) * 7.5
    afstand_mm = (147.58 * 25) / object_mm_on_sensor
    return afstand_mm
def scalepoints(points):
    point1 = points[0]
    point2 = points[1]
    distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2) * 0.56
    point3 = [point2[0], point2[1] - distance]
    point4 = [point1[0], point1[1] - distance]
    return np.array([point1, point2, point3, point4], np.int32)
def offset_points(points, depth):

    
    offset_points = points.copy()
    
    # Calculate the center of the front face
    center = np.mean(points, axis=0)
    
    # calculate perspective direction
    perspective_angle_X=(angle_between_points([points[0],points[1]])/45)*1
    perspective_angle_Y=0.02 # Y camera angle does not change
    perspective_direction = np.array([perspective_angle_X,perspective_angle_Y])
    print(f"perspective_direction X:{perspective_angle_X}")
    
    # Apply the offset with perspective to each point
    for i in range(len(points)):
        # Calculate vector from point to center
        vector = points[i] - center
        
        # Calculate distance from center
        distance = np.linalg.norm(vector)
        
        # Apply the offset with perspective adjustment
        if distance > 0:
            # The offset is based on the point's position relative to center
            offset_points[i] = points[i] + perspective_direction * depth
    
    return offset_points.astype(np.int32)

def draw_cube(image, front_face):
    """
    Draws a 3D cube on the image given the front face points
    
    Args:
        image: The image to draw on
        front_face: np.array of 4 points defining the front face
    
    Returns:
        Image with cube drawn on it
    """
    #sccale van de deapth beasd from side line
    depth= np.sqrt((front_face[1][0] - front_face[0][0])**2 + (front_face[1][1] - front_face[0][1])**2) * -0.79 #is negative becose we want the offset

    # Make a copy of the image to draw on
    result = image.copy()
    
    # Draw the front face
    cv2.polylines(result, [front_face], isClosed=True, color=(0, 255, 0), thickness=3)
    
    # Calculate the back face
    back_face = offset_points(front_face, depth)
    
    # Draw the back face
    cv2.polylines(result, [back_face], isClosed=True, color=(0, 0, 255), thickness=3)
    
    # Draw the distance text on the image
    cv2.putText(
        result, 
        f"Distance: {cal_distance(front_face[1][1] - front_face[2][1]):.2f} mm", 
        (front_face[0][0], front_face[0][1] - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        1, 
        (255,255,255), 
        2
    )
    # Draw the connecting edges
    for i in range(4):
        cv2.line(result, 
                tuple(front_face[i]), 
                tuple(back_face[i]),
                color=(255, 0, 0), thickness=3)
    
    return result
# Global variables
point = False
# point1,2 are only global for testing purpuses
point1 = [0, 0]
point2 = [0, 0]
image = None
display_image = None

#for testing points are set by mouse
def mouse_callback(event, x, y, flags, param):
    global point, point1, point2, image, display_image
    
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Mouse clicked at: x={x}, y={y}")
        
        if not point:
            point = True
            print("First point selected")
            point1 = [x, y]
        else:
            point = False
            print("Second point selected")
            point2 = [x, y]
            
            # Create the front face from the two points
            front_face = scalepoints([point1, point2])
            angle_between_points([point1, point2])            
            # Reload the original image
            display_image = image.copy()
            
            # Draw the 3D cube
            display_image = draw_cube(display_image, front_face)
            
            # Display the image
            cv2.imshow('3D Cube on Image', display_image)

# Main code
def main():
    global image, display_image
    
    # Load an image
    image = cv2.imread('35ddisejetracer.jpeg')  # test image here 
    # Resize the image to 1000x700
    image = cv2.resize(image, (1000, 700))
    # Check if the image was loaded successfully
    if image is None:
        print("Error: Could not load image.")
        exit()
    
    # Make a copy for display
    display_image = image.copy()
    
    # Create window and set mouse callback
    cv2.namedWindow('3D Cube on Image')
    cv2.setMouseCallback('3D Cube on Image', mouse_callback)
    
    # Display the image
    cv2.imshow('3D Cube on Image', display_image)
    
    # Wait for key press and close windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()