import numpy as np
import math
import cv2

EPSILON = .00001
UPSILON = 150000

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


        if depth < 0:
            return [['depth < 0'], t]

        if depth == 0:
            # shadows
            # vector pointing to light
            shadow_ray = [x, y, z, Lx, Ly, Lz]
            for object in scene:
                if isinstance(object, Plane):
                    continue

                found = object.find_intersection(shadow_ray, -1)
                ttk = found[-1]
                if ttk < float("inf") and ttk > EPSILON: 
                    return [[object.mirror,object.mirror,object.mirror], 0]

            return self.diffuse_shading(x, y, z, Lx, Ly, Lz, t)

        # dont reflect if no mirror
        if self.mirror == 0:
            return self.diffuse_shading(x, y, z, Lx, Ly, Lz, t)

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
                test = [[0,0,0], 1]
            ttk = found[1]
            testval = abs(test[1])

            if ttk < float("inf") and ttk > EPSILON and testval < float("inf") and testval > EPSILON: # very very bad workaround for unknown issue TODO FIX
                return [np.array(object.find_intersection([cx,cy,cz, reflection_ray[0], reflection_ray[1], reflection_ray[2]], depth-1)[0]) * (self.mirror/100) + (1 - (self.mirror/100)) * np.array(self.diffuse_shading(x, y, z, Lx, Ly, Lz, t)[0]), ttk]

        return [[255,255,255], 0]

    def diffuse_shading(self, x,y,z,Lx,Ly,Lz,t):
        cx,cy,cz = self.pos[0],self.pos[1],self.pos[2]
        R = self.r
        # diffuse shading
        N = [(x - cx)/R, (y - cy)/R, (z - cz)/R] # unit normal vector
        L = unit_vector(np.array([Lx-x, Ly-y, Lz-z]))

        fctr = math.cos(angle_between(N, L))
        kd = .3
        ka = .3
        d = 1 / math.sqrt((Lx-x)**2 + (Ly-y)**2 + (Lz-z)**2) ** 2
        d *= UPSILON
        if d > 1:
            d = 1
        return [d * np.array(ka*np.array(self.color) + kd * np.array([fctr * self.color[0], fctr * self.color[1], fctr * self.color[2]])), t]
            
        #return [ka*np.array(self.color) + kd * np.array([fctr * self.color[0], fctr * self.color[1], fctr * self.color[2]]), t]
       
class Plane:
    def __init__(self, zpos, mirror=0):
        self.zpos = zpos
        self.mirror = mirror
        self.ground = np.array([255,255,255])
        self.sky = np.array([235, 206, 135])

    def diffuse_shading(self, x,y,z, t):
        Lx, Ly, Lz = lights[0].pos[0],lights[0].pos[1],lights[0].pos[2]
        # inverse square
        d = 1 / math.sqrt((Lx-x)**2 + (Ly-y)**2 + (Lz-z)**2) ** 2
        d *= UPSILON

        if d > 1:
            d = 1
        
    
        # checker pattern
        checker_size = 100
        if (int(x/checker_size) + int(y/checker_size)) % 2 == 0:
            return [d * np.array([0,0,180]),t]
        else:
            return [d * np.array([0,0,0]), t]


    def find_intersection(self,ray, depth=1):
        Lx, Ly, Lz = lights[0].pos[0],lights[0].pos[1],lights[0].pos[2]
        if ray[5] < 0:#self.zpos:
            # shadows
            # vector pointing to light
            if ray[5] == 0:
                 [(100,0,100), 100000]

            # calculate ray intersection point
            t = (self.zpos-ray[2])/ray[5]
            x = ray[3] * t + ray[0]
            y = ray[4] * t + ray[1]
            z = ray[5] * t + ray[2]

            # create ray pointing towards light to see if obstructed
            shadow_ray = [x, y, z, Lx, Ly, Lz]
            for object in scene:
                if object == self:
                    break
                found = object.find_intersection(shadow_ray, 0)
                ttk = found[-1]
                if ttk < float("inf") and ttk > EPSILON:
                    return [[0,0,0], 100000]
                 
            N = unit_vector(np.array([x,y,z])) # unit normal vector
            unit_ray = unit_vector(np.array([x - ray[0], y - ray[1], z - ray[2]]))
            reflection_ray = unit_ray - 2 * (np.dot(N, unit_ray)) * N
            
            if depth == 0:
                return self.diffuse_shading(x,y,z,t)
                
            for objects in scene:
                found = objects.find_intersection([x,y,z, reflection_ray[0], reflection_ray[1], reflection_ray[2]], depth-1)
                ttk = found[-1]
                
                if ttk < float("inf") and ttk > EPSILON:
                    return [np.array(object.find_intersection([x,y,z, reflection_ray[0], reflection_ray[1], reflection_ray[2]], depth-1)[0]) * (self.mirror/100) + (1 - (self.mirror/100)) * np.array(self.diffuse_shading(x, y, z, t)[0]), ttk]

            return [self.diffuse_shading(x,y,z,t)[0], 1000000]


        else: # sky
            gradient = 0
            if gradient:
                if ray[5] == 0:
                    return [(235, 206, 135), 100000]

                t = abs((self.zpos -ray[2])/ray[5])
                t /= 1
                if t > 255:
                    return [self.sky, 10000]
                return [self.sky * (t/255),10000]
            else:
                # calculate ray intersection point
                t = (700-ray[2])/(ray[5] + .001)
                x = ray[3] * t + ray[0]
                y = ray[4] * t + ray[1]
                z = ray[5] * t + ray[2]

                d = 1 / math.sqrt((Lx-x)**2 + (Ly-y)**2 + (Lz-z)**2) ** 2
                d *= UPSILON
                if d > 1:
                    d = 1
                if d < 0:
                    d = 0

                try:
                    checker_size = 100
                    if (int(x/checker_size) + int(y/checker_size)) % 2 == 0:
                        return [d * self.sky,100000]
                    else:
                        return [d * self.sky * 2, 100000]
                except:
                    return [[0,0,0], 100000]


class Light:
    def __init__(self, pos):
        self.pos = pos

precision = 3

l = 200 # put a comment here
w = 200
fov = 100
screen = []

camera = [l/2,-50,w/2 + 10]

scene = [
    Sphere([150,50,150], 15, [192,192,192], 100),
    Sphere([100,70,50], 30, [192,192,192], 0),
    Sphere([100,90,130], 30, [255,0,0], 50),
    Sphere([50,85,80], 30, [0,255,0], 50),
    Sphere([150,85,80], 30, [0,0,255], 50),
    
    Plane(0, 100)
]

lights = [Light([100, 50, 200])]
#
for x in range(-int(w* precision)//2,int(w* precision)//2):
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
            found = object.find_intersection(ray,1)
            if found[-1] < closest:
                closest = found[-1]
                color = found[0]

        row.append(color)
                
    screen.append(row)
    
    print(str(round(((x + 100) / 2), 2)) + "%")
    src = np.rot90(np.uint8(screen))
    cv2.imshow('image', src)
    cv2.waitKey(1)

print("100.0%")
cv2.imwrite('render.png',src)
cv2.imshow('image',src)
cv2.waitKey(0)

  
