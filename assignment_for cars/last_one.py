import cv2
import numpy as np

def resize_and_position_mask(mask, target_size, x_offset=0, y_offset=0, scale_factor=1.0):
    """
    Resize mask and position it with custom offsets.
    """
    target_h, target_w = target_size
    mask_h, mask_w = mask.shape[:2]
    
    # Calculate new dimensions based on scale factor
    new_w = int(mask_w * scale_factor)
    new_h = int(mask_h * scale_factor)
    
    # Resize the mask
    resized_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create a blank canvas of target size
    final_mask = np.zeros((target_h, target_w), dtype=np.uint8)
    
    # Calculate actual placement coordinates
    y_start = y_offset
    y_end = y_offset + new_h
    x_start = x_offset
    x_end = x_offset + new_w
    
    # Handle cases where the mask goes outside the image bounds
    if x_start < 0:
        resized_mask = resized_mask[:, -x_start:]
        x_start = 0
    if y_start < 0:
        resized_mask = resized_mask[-y_start:, :]
        y_start = 0
    if x_end > target_w:
        resized_mask = resized_mask[:, :-(x_end - target_w)]
        x_end = target_w
    if y_end > target_h:
        resized_mask = resized_mask[:-(y_end - target_h), :]
        y_end = target_h
    
    # Place the mask at the specified position
    try:
        final_mask[y_start:y_end, x_start:x_end] = resized_mask
    except ValueError as e:
        print(f"Warning: Mask placement error. Adjusting bounds. Error: {e}")
        h, w = resized_mask.shape
        y_end = min(y_start + h, target_h)
        x_end = min(x_start + w, target_w)
        final_mask[y_start:y_end, x_start:x_end] = resized_mask[:y_end-y_start, :x_end-x_start]
    
    return final_mask

def create_background(floor_path, wall_path, target_size):
    """
    Create a composite background from floor and wall images.
    """
    # Read floor and wall images
    floor = cv2.imread(floor_path)
    wall = cv2.imread(wall_path)
    
    if floor is None or wall is None:
        raise ValueError("Failed to load floor or wall image")
    
    # Resize images to match target size
    floor = cv2.resize(floor, (target_size[1], target_size[0]))
    wall = cv2.resize(wall, (target_size[1], target_size[0]))
    
    # Create a mask for splitting the image (horizontal split at 40% from top)
    split_point = int(target_size[0] * 0.6)  # Adjust this value to change wall/floor ratio
    mask = np.zeros(target_size[:2], dtype=np.uint8)
    mask[split_point:, :] = 255
    
    # Combine wall and floor
    background = wall.copy()
    background[split_point:, :] = floor[split_point:, :]
    
    return background

def process_images(car_image_path, car_mask_path, shadow_mask_path, 
                  floor_path, wall_path, 
                  shadow_x_offset=0, shadow_y_offset=0, shadow_scale=1.0):
    """
    Process all images to create the final composition.
    """
    # Read car image and masks
    car_image = cv2.imread(car_image_path)
    car_mask = cv2.imread(car_mask_path, cv2.IMREAD_GRAYSCALE)
    shadow_mask = cv2.imread(shadow_mask_path, cv2.IMREAD_GRAYSCALE)
    
    if car_image is None or car_mask is None or shadow_mask is None:
        raise ValueError("Failed to load one or more images")
    
    # Get target size from original image
    target_size = car_image.shape
    
    # Create composite background
    background = create_background(floor_path, wall_path, target_size)
    
    # Process car mask (clean up noise using morphological operations)
    car_mask = cv2.resize(car_mask, (target_size[1], target_size[0]))
    kernel = np.ones((3,3), np.uint8)
    car_mask = cv2.morphologyEx(car_mask, cv2.MORPH_CLOSE, kernel)
    car_mask = cv2.morphologyEx(car_mask, cv2.MORPH_OPEN, kernel)
    _, car_mask = cv2.threshold(car_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Position and process shadow mask
    shadow_mask = resize_and_position_mask(
        shadow_mask, 
        target_size[:2],
        x_offset=shadow_x_offset,
        y_offset=shadow_y_offset,
        scale_factor=shadow_scale
    )
    _, shadow_mask = cv2.threshold(shadow_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Create inverse masks
    car_mask_inv = cv2.bitwise_not(car_mask)
    shadow_mask_inv = cv2.bitwise_not(shadow_mask)
    
    # Extract car from original image
    masked_car = cv2.bitwise_and(car_image, car_image, mask=car_mask)
    
    # Create shadowed background
    shadow_strength = 0.4  # Adjust this value to control shadow intensity
    background_with_shadow = background.copy()
    background_with_shadow[shadow_mask > 0] = background_with_shadow[shadow_mask > 0] * (1 - shadow_strength)
    
    # Combine background with car
    background_masked = cv2.bitwise_and(background_with_shadow, background_with_shadow, mask=car_mask_inv)
    final_image = cv2.add(masked_car, background_masked)
    
    return final_image

def main():
    # Image paths
    car_image_path = "/home/multi-lap-49/Downloads/Assignments/assignment/images/6.jpeg"
    car_mask_path = "/home/multi-lap-49/Downloads/Assignments/assignment/car_masks/6.png"
    shadow_mask_path = "/home/multi-lap-49/Downloads/Assignments/assignment/shadow_masks/6.png"
    floor_path = "/home/multi-lap-49/Downloads/Assignments/assignment/floor.png"
    wall_path = "/home/multi-lap-49/Downloads/Assignments/assignment/wall.png"
    
    # Adjust these values for shadow positioning
    shadow_x_offset = 510  # Adjust based on your needs
    shadow_y_offset = 590   # Adjust based on your needs
    shadow_scale = 1.0     # Adjust based on your needs
    
    try:
        # Process images
        result = process_images(
            car_image_path, 
            car_mask_path, 
            shadow_mask_path,
            floor_path,
            wall_path,
            shadow_x_offset=shadow_x_offset,
            shadow_y_offset=shadow_y_offset,
            shadow_scale=shadow_scale
        )
        
        # Save the result
        cv2.imwrite("final_composition.jpg", result)
        
        # Display the result (optional)
        cv2.imshow("Final Composition", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()