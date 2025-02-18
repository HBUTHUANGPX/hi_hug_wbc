import pygame

pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Test Keyboard Events")

running = True
while running:
    for event in pygame.event.get():
        print(event)  # 打印所有事件
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

pygame.quit()
