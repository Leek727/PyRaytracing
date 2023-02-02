import numpy as np
import math
import cv2


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

class Sphere:
    def __init__(self, pos, r, color=[255,255,255], mirror=100):
        self.pos = pos
        self.r = r
        self.mirror = mirror
        self.color = color

    def find_light(self, start_pos, light_pos):
        """
        Casts a ray from a specific point on the sphere to the light source and check obstructions
        """
        ray = [start_pos[0], start_pos[1], start_pos[2], light_pos[0] - start_pos[0], light_pos[1] - start_pos[1], light_pos[2] - start_pos[2]]
        for object in scene:
            
            try:
                if object.find_intersect(ray) > 0:
                    print(object.find_intersect(ray))
                    return -1

            except Exception as e:
                print(e)
        else:
            return 1

    def find_intersection(self, ray, depth=1):
        x0, y0, z0 = ray[0], ray[1], ray[2]
        x1, y1, z1 = ray[0] + ray[3], ray[1] + ray[4], ray[2] + ray[5]
        cx,cy,cz = self.pos[0],self.pos[1],self.pos[2]
        R = self.r

        dx = x1 - x0
        dy = y1 - y0
        dz = z1 - z0
        a = dx*dx + dy*dy + dz*dz
        b = 2*dx*(x0-cx) + 2*dy*(y0-cy) + 2*dz*(z0-cz)
        c = cx*cx + cy*cy + cz*cz + x0*x0 + y0*y0 + z0*z0 + -2*(cx*x0 + cy*y0 + cz*z0) - R*R
        
        disc = b**2 - 4*a*c
        if disc <= 0:
            return [[0,0,0],float("inf")]

        # find ray time
        t = (-b - math.sqrt(disc))/(2*a)

        # check if depth reached
        if depth == 0:
            return [self.color, t]


        # find point on sphere
        x = x0 + t*dx
        y = y0 + t*dy
        z = z0 + t*dz

        # light source coords
        Lx, Ly, Lz = lights[0].pos[0],lights[0].pos[1],lights[0].pos[2]

        # shadows
        # vector pointing to light
        shadow_ray = [x, y, z, Lx, Ly, Lz]
        for object in scene:
            found = object.find_intersection(shadow_ray, 0)
            ttk = found[-1]
            if ttk < float("inf") and ttk > .0001:           
                return [[0,0,0], 0]

        # add reflections
        # vector reflected off objects
        return self.diffuse_shading(x, y, z, Lx, Ly, Lz,t)
        

    def diffuse_shading(self, x,y,z,Lx,Ly,Lz,t):
        cx,cy,cz = self.pos[0],self.pos[1],self.pos[2]
        R = self.r
        # diffuse shading
        N = [(x - cx)/R, (y - cy)/R, (z - cz)/R] # unit normal vector
        L = unit_vector(np.array([Lx-x, Ly-y, Lz-z]))

        fctr = math.cos(angle_between(N, L))
        kd = .5
        ka = .5

        return [ka*np.array(self.color) + kd * np.array([fctr * self.color[0], fctr * self.color[1], fctr * self.color[2]]), t]
       
class Plane:
    def __init__(self, zpos):
        self.zpos = zpos

    def find_intersection(self,ray, depth=1):
        if depth == 0:
            return [[0,0,0], float("inf")]

        if ray[5] < self.zpos:
            # shadows
            # light source coords
            Lx, Ly, Lz = lights[0].pos[0],lights[0].pos[1],lights[0].pos[2]

            # vector pointing to light
            if ray[5] == 0:
                 [(100,0,100), 100000]

            t = (self.zpos-ray[2])/ray[5]
            x = ray[3] * t + ray[0]
            y = ray[4] * t + ray[1]
            z = ray[5] * t + ray[2]

            shadow_ray = [x, y, z, Lx, Ly, Lz]
            for object in scene:
                found = object.find_intersection(shadow_ray, 0)
                ttk = found[-1]
                if ttk < float("inf") and ttk > 0:           
                    return [[0,0,0], float("inf")]


            return [(83,118,155), 100000]

        else:
            return [(235, 206, 135), 100000]

class Light:
    def __init__(self, pos):
        self.pos = pos


l = 200
w = 200
fov = 100
screen = []

camera = [l/2,-50,w/2]


light_coord = [200, 40, 200]

scene = [Sphere([l/3,160,170], 40, [255,0,0]), Sphere([30,90,80], 40, [0,255,0]),  Sphere([150,90,40], 40, [0,0,255]), Sphere(light_coord, 5, [255,255,255]), Plane(0)]

lights = [Light(light_coord)]
for x in range(-w//2,w//2):
    row = []
    for z in range(-l//2, l//2):
        #row.append([100,100,0])
        # camera generate rays
        #ray = [camera[0] + x, camera[1], camera[2] + z, 0,1,0]
        ray = [camera[0], camera[1], camera[2], x/fov, 1 , z/ fov]
        #ray = [camera[0], camera[1], camera[2], 0, 1, 0]
        color = [0,0,0]
        closest = float("inf")
        for object in scene:
            found = object.find_intersection(ray)
            if found[-1] < closest:
                closest = found[-1]
                color = found[0]


        row.append(color)
                
    screen.append(row)
    
    cv2.imshow('image', np.rot90(np.uint8(screen)))
    cv2.waitKey(1)

cv2.imshow('image', np.rot90(np.uint8(screen)))
cv2.waitKey(0)