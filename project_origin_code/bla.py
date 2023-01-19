# importing pygame module
import pygame

# importing sys module
import sys

# initialising pygame
pygame.init()
angle = 0

# creating display
display = pygame.display.set_mode((300, 300))

# creating a running loop
while True:
	
	# creating a loop to check events that
	# are occurring
    
	
    for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            
            # checking if keydown event happened or not
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    print("Key ESCAPE has been pressed")
                    pygame.quit()
                    sys.exit()

                if event.key == pygame.K_a:
                    print("Key a has been pressed")
                    angle += 1
                    print(angle)
                            
                if event.key == pygame.K_d:
                    print("Key d has been pressed")
                    angle -= 1
                    print(angle)

    
