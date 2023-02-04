import numpy as np
import math
import cv2

EPSILON = .000001

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. """
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
                    return -1

            except Exception as e:
                print(e)
        else:
            return 1

    def find_intersection(self, ray, depth=1):
        # set calculation variables
        x0, y0, z0 = ray[0], ray[1], ray[2]
        x1, y1, z1 = ray[0] + ray[3], ray[1] + ray[4], ray[2] + ray[5]
        cx,cy,cz = self.pos[0],self.pos[1],self.pos[2]
        R = self.r

        # solve quadratic equation to find intersection times
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

        # find point on sphere
        x = x0 + t*dx
        y = y0 + t*dy
        z = z0 + t*dz

        # light source coords
        Lx, Ly, Lz = lights[0].pos[0],lights[0].pos[1],lights[0].pos[2]

        # check if depth reached
        if depth == 0:
            return self.diffuse_shading(x, y, z, Lx, Ly, Lz, t)
        
        if depth < 0:
            return [[], t]
        # dont reflect if no mirror
        if self.mirror == 0:
            return self.diffuse_shading(x, y, z, Lx, Ly, Lz, t)

        if depth == 0:
            # shadows
            # vector pointing to light
            shadow_ray = [x, y, z, Lx, Ly, Lz]
            for object in scene:
                if isinstance(object, Plane):
                    continue

                found = object.find_intersection(shadow_ray, 0)
                ttk = found[-1]
                if ttk < float("inf") and ttk > EPSILON:           
                    return [[object.mirror,object.mirror,object.mirror], 0]

        # reflections
        # calculate normal vector and reflect ray d−2(d⋅n)n
        N = np.array([(x - cx)/R, (y - cy)/R, (z - cz)/R]) # unit normal vector
        unit_ray = unit_vector(np.array([x - ray[0], y - ray[1], z - ray[2]]))
        reflection_ray = unit_ray - 2 * (np.dot(N, unit_ray)) * N

        for object in scene:
            if object == self:
                continue
            found = object.find_intersection([cx,cy,cz, reflection_ray[0], reflection_ray[1], reflection_ray[2]], depth-1)
            if not isinstance(object, Plane):
                uaa = unit_vector(np.array([object.pos[0] - x, object.pos[1] - y, object.pos[2] - z]))
                test = self.find_intersection([x,y,z, uaa[0], uaa[1], uaa[2]], -1)
            else:
                test = [[], 1]
            ttk = found[1]
            testval = abs(test[1])

            if ttk < float("inf") and ttk > EPSILON and testval < float("inf") and testval > EPSILON: # very very bad workaround for unknown issue TODO FIX
                if depth == 0:
                    return [np.array(found[0]) * (self.mirror/100) + (1 - (self.mirror/100)) * np.array(self.diffuse_shading(x, y, z, Lx, Ly, Lz, t)[0]), ttk]
                else:
                    return [np.array(object.find_intersection([cx,cy,cz, reflection_ray[0], reflection_ray[1], reflection_ray[2]], depth-1)[0]) * (self.mirror/100) + (1 - (self.mirror/100)) * np.array(self.diffuse_shading(x, y, z, Lx, Ly, Lz, t)[0]), ttk]

        return [[255,255,255], 0]

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
    def __init__(self, zpos, mirror=0):
        self.zpos = zpos
        self.ground = np.array([255,255,255])
        self.sky = np.array([235, 206, 135])


    def find_intersection(self,ray, depth=1):
        if ray[5] < 0:#self.zpos:
            # shadows
            # light source coords
            Lx, Ly, Lz = lights[0].pos[0],lights[0].pos[1],lights[0].pos[2]

            # vector pointing to light
            if ray[5] == 0:
                 [(100,0,100), 100000]

            # calculate ray intersecetion point
            t = (self.zpos-ray[2])/ray[5]
            x = ray[3] * t + ray[0]
            y = ray[4] * t + ray[1]
            z = ray[5] * t + ray[2]

            # create ray pointing towards light to see if obstructed
            shadow_ray = [x, y, z, Lx, Ly, Lz]
            color = []
            for object in scene:
                if object == self:
                    break
                found = object.find_intersection(shadow_ray, 0)
                ttk = found[-1]
                if ttk < float("inf") and ttk > EPSILON:
                    opaquenessness = (object.mirror / 100)
                    return [[0,0,0], 100000]
                    #color = [opaquenessness * np.array(self.ground)]

            checker_size = 100
            if (int(x/checker_size) + int(y/checker_size)) % 2 == 0:
                return [[0,0,180],100000]
            else:
                return [[0,0,0], 100000]

        else:
            if ray[5] == 0:
                return [(235, 206, 135), 100000]

            t = abs((self.zpos -ray[2])/ray[5])
            t /= 4
            if t > 255:
                return [self.sky, 10000]
            return [self.sky * (t/255),10000]

class Light:
    def __init__(self, pos):
        self.pos = pos

precision = 1.5

l = 200 # put a comment here
w = 200
fov = 100
screen = []

camera = [l/2,-50,w/2 + 10]


light_coord = [100, 50, 200]

scene = [Sphere([100,100,130], 50, [255,255,0], 100), Sphere([50,50,80], 30, [255,255,0], 100), Sphere([150,50,80], 30, [255,255,0], 0),  Plane(0)]

lights = [Light(light_coord)]
for x in range(-int(w* precision)//2 ,int(w* precision)//2):
    x /= precision
    row = []
    for z in range(-int(l * precision)//2, int(l * precision)//2):
        z /= precision
        #row.append([100,100,0])
        # camera generate rays
        #ray = [camera[0] + x, camera[1], camera[2] + z, 0,1,0]
        ray = [camera[0], camera[1], camera[2], x/fov, 1 , z/ fov]
        #ray = [camera[0], camera[1], camera[2], 0, 1, 0]
        color = [0,0,0]
        closest = float("inf")
        for object in scene:
            found = object.find_intersection(ray,2)
            if found[-1] < closest:
                closest = found[-1]
                color = found[0]

        row.append(color)
                
    screen.append(row)
    
    cv2.imshow('image', np.rot90(np.uint8(screen)))
    cv2.waitKey(1)

cv2.imwrite('render.png', np.rot90(np.uint8(screen)))
cv2.imshow('image', np.rot90(np.uint8(screen)))
cv2.waitKey(0)
