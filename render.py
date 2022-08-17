import sys
from cmath import nan
from image import Image, Color
from model import Model
from shape import Point, Line, Triangle
from vector import Vector
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import radians, cos, sin, tan

def getVertexNormal(vertIndex, faceNormalsByVertex):
	# Compute vertex normals by averaging the normals of adjacent faces
	normal = Vector(0, 0, 0)
	for adjNormal in faceNormalsByVertex[vertIndex]:
		normal = normal + adjNormal

	return normal / len(faceNormalsByVertex[vertIndex])


# translation matrix
def translate(dx, dy, dz):
	translate_matrix = np.matrix([
		[1,	  	0, 		0, 		dx],
		[0,		1,     	0, 		dy],
		[0, 	0,  	1, 		dz],
		[0, 	0, 		0, 		1]
	], dtype=float)
	return translate_matrix

# rotation matrices
def rotate_x(degrees):
	theta = radians(degrees)

	rot_matrix = np.matrix([
		[1,	  	0, 				0, 					0],
		[0,		cos(theta),     -sin(theta), 		0],
		[0, 	sin(theta),  	cos(theta), 		0],
		[0, 	0, 				0, 					1]
	], dtype=float)

	return rot_matrix

def rotate_y(degrees):
	theta = radians(degrees)

	rot_matrix = np.matrix([
		[cos(theta), 	0, 		sin(theta), 		0],
		[0,				1,     	0, 					0],
		[-sin(theta), 	0,  	cos(theta), 		0],
		[0,			 	0, 		0, 					1]
	], dtype=float)

	return rot_matrix	

def rotate_z(degrees):
	theta = radians(degrees)

	rot_matrix = np.matrix([
		[cos(theta), 	-sin(theta), 	0, 			0],
		[sin(theta),	cos(theta),		0,      	0],
		[0, 			0,  	 		1, 			0],
		[0,			 	0, 				0, 			1]
	], dtype=float)
	return rot_matrix	
	
# scaling matrix
def scale(sx, sy, sz):
	scale_matrix = np.matrix([
		[sx,	0, 		0, 		0],
		[0,		sy,    	0, 		0],
		[0, 	0,  	sz, 	0],
		[0, 	0, 		0, 		1]
	], dtype=float)
	return scale_matrix

# orthographic projection 
def getOrthographicProjection(x, y, z):
	# Convert vertex from world space to screen space
	# by dropping the z-coordinate (Orthographic projection)
	screenX = int((x+1.0)*width/2.0)
	screenY = int((y+1.0)*height/2.0)

	return screenX, screenY

# helper function for normalisation
def norm(v):
    m = np.sqrt(np.sum(v ** 2))
    if m == 0:
        return v
    return v / m


# get view transform
def getViewTransform():
	eye = np.array([
		[0],
		[0],
		[-3]
		], dtype=float)
	center = np.array([
		[0],
		[0],
		[10]
		], dtype=float)
	up = np.array([
		[0],
		[1],
		[-3]
		], dtype=float)
	up = norm(up)
	o = eye-center
	cam_z = norm(o) 
	v_i = np.cross(up, cam_z, axis=0)
	cam_x = norm(v_i)
	cam_y = np.cross(cam_z, cam_x, axis=0)

	r = np.matrix([
		[cam_x[0], cam_x[1], cam_x[2], 0],
		[cam_y[0], cam_y[1], cam_y[2], 0],
		[cam_z[0], cam_z[1], cam_z[2], 0],
		[0,0,0, 1]
	], dtype=float	)

	t = np.matrix([
		[1, 0, 0, -(eye[0])],
		[0, 1, 0, -(eye[1])],
		[0, 0, 1, -(eye[2])],
		[0, 0, 0, 1],
	], dtype=float)

	V = np.dot(r, t)
	return V
# perspective projection
def getPerspectiveProjection(V, M, x, y, z, n):

	# vertex vector
	p = np.array([
		[x],
		[y], 
		[z], 
		[1]], dtype=float)

	# normal vector
	n_ = np.array([
		[n.x],
		[n.y], 
		[n.z], 
	], dtype=float)
	
	# perspective 
	near = 250.0
	far = 1000.0
	top = height / 2
	bottom = -(height / 2)
	left = -(width /2 )
	right = width/2
	r_l = right-left
	t_b = top-bottom
	P = np.matrix([
		[(2*near)/r_l, 0, (right + left)/r_l, 0],
		[0, 2*near/t_b, (top+bottom)/(t_b), 0],
		[0, 0, -(far+near)/(far-near), -(2*far*near)/(far-near)],
		[0, 0, -1, 0],
	], dtype=float)

	clip = P * V * p

	n_ = (np.linalg.inv(V[:3, :3]) * np.linalg.inv(M)).T * n_

	# perspective divide
	D = np.array([
		[clip[0,0]/clip[3,0]],
		[clip[1,0]/clip[3,0]], 
		[clip[2,0] /clip[3,0]], 
		[1]
		], dtype=float)

	# window transform 
	screenX = int((D[0]+1) * (width/2.0))
	screenY = int((D[1]+1) * (height/2.0))
	screenZ = int((D[2]) * (1/2.0) )

	return screenX, screenY, screenZ, n_	


# function that gets conjugate of a given quaternion 
def quaternion_conjugation(q):
	return [q[0], -q[1], -q[2], -q[3]]

# quaternion multiplication of two quaternions
def quaternion_product(a, b):
	return np.array([(a[0] * b[0]) - (a[1] * b[1]) - (a[2] * b[2]) - (a[3] * b[3]),
					(a[0] * b[1]) + (b[0]* a[1] ) - (a[2] * b[3]) + (b[2] * a[3]),
					(a[0] * b[2]) + (b[3] * a[1]) + (b[0] * a[2]) - (a[3] * b[1]),
					(a[0] * b[3]) - (b[2] * a[1]) + (a[2] * b[1]) + (b[0] * a[3])], dtype=np.float64)

# function ton convert euler angle to quaternion
def euler_to_quaternion(pitch, yaw, roll):
	cos_pitch = np.cos(pitch / 2)
	cos_yaw = np.cos(yaw / 2)
	cos_roll = np.cos(roll / 2)
	sin_pitch = np.sin(pitch / 2)
	sin_yaw = np.sin(yaw / 2)
	sin_roll = np.sin(roll / 2)

	q0 = (cos_pitch * cos_yaw * cos_roll) + (sin_pitch * sin_yaw * sin_roll)
	q1 = (cos_pitch * cos_yaw * sin_roll) - (sin_pitch * sin_yaw * cos_roll)
	q2 = (sin_pitch * cos_yaw * cos_roll) + (cos_pitch * sin_yaw * sin_roll)
	q3 = (cos_pitch * sin_yaw * cos_roll) - (sin_pitch * cos_yaw * sin_roll)

	return [q0, q1, q2, q3]

# function to convert quaternion to euler angle
def compute_euler_angle(q0, q1, q2, q3):
	
	# pitch
	sin_pitch = 2 * ((q0 * q2) - (q1 * q3))
	if abs(sin_pitch) >= 1:
		if sin_pitch < 0:
			pitch = -np.pi/2
		else:
			pitch = np.pi/2
	else:
		pitch = np.arcsin(sin_pitch)

	# yaw
	sin_yaw_cos_pitch = 2 * ((q0 * q3) + (q1*q2))
	cos_yaw_cos_pitch = 1 - 2 * (q2**2 + q3**2)

	yaw=np.arctan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)

	# roll
	sin_roll_cos_pitch=2 * ((q0 * q1) + (q2*q3))
	cos_roll_cos_pitch=1 - 2 * (q1**2 + q2**2)
	roll=np.arctan2(sin_roll_cos_pitch, cos_roll_cos_pitch)

	return pitch, yaw, roll

# function to perform data normalisation and rotation units
def convert_data(df):

	# function for normalising accelerometer and magnetometer values
	def normalise_values(row, axis, value_type):
		if value_type == "a":
			X = row[" accelerometer.X"]
			Y = row[" accelerometer.X"]
			Z = row[" accelerometer.Z"]
		else:
			X = row[" magnetometer.X"]
			Y = row[" magnetometer.X"]
			Z = row[" magnetometer.Z "]

		mag = np.sqrt((X ** 2) + (Y ** 2) + (Z ** 2))

		if mag == 0 or mag == nan:
			if axis == 'x':
				return X
			elif axis == 'y':
				return Y
			else: 
				return Z 
		else:
			if axis == 'x':
				return X/mag
			elif axis == 'y':
				return Y/mag
			else: 
				return Z/mag

	# convert rotational rates to rads/s 
	df[" gyroscope.X"] = df[" gyroscope.X"].apply(lambda x: x * 0.017453)
	df[" gyroscope.Y"] = df[" gyroscope.Y"].apply(lambda x: x * 0.017453)
	df[" gyroscope.Z"] = df[" gyroscope.Z"].apply(lambda x: x * 0.017453)

	# normalise accel. and magnetometer magnitude
	df[" accelerometer.X"] = df.apply(normalise_values, args=("x", "a") ,axis=1)
	df[" accelerometer.Y"] = df.apply(normalise_values, args=("y", "a") ,axis=1)
	df[" accelerometer.Z"] = df.apply(normalise_values, args=("z", "a"), axis=1)

	df[" magnetometer.X"] = df.apply(normalise_values, args=("x", "m") ,axis=1)
	df[" magnetometer.Y"] = df.apply(normalise_values, args=("y", "m") ,axis=1)
	df[" magnetometer.Z "] = df.apply(normalise_values, args=("z", "m"), axis=1)
	return df 

# function to get quaternion given axis-angle representation
def get_quaternion(axis, angle):
	# previous implementation
	q0 = np.cos(angle/2)
	q1 = axis[0 ,0] * np.sin(angle/2)    
	q2 = axis[1, 0] * np.sin(angle/2)
	q3 = axis[2, 0] * np.sin(angle/2)

	return [q0, q1, q2, q3]

# retrieve axis 
def getAxis(q0, q1, q2, q3):
	rot = 2*np.arccos(q0)
	if q0 == 1:
		return 1, 0, 0
	else:
		
		v = [(q1/np.sin(rot/2)), (q2/np.sin(rot/2)), (q3/np.sin(rot/2))]
		return v[0], v[1], v[2]

# function to get global acceleration from local space
def get_world_accel(a,q, q_inv):
	return quaternion_product(quaternion_product(q_inv, a), q)

def complementary_filter(data, previous_obs, mode, mag_data=None, sample_rate=256):
	# simple dead reckoning filter to estimate orientation via gyroscopic measurements
	def drf(x,y,z):
		
		# compute magnitude of the quaternion (?)
		magnitude = np.sqrt((x**2) + (y**2) +(z**2) )
		
		# compute instantaneous axis of rotation
		rotation_axis = np.array([
			[x],
			[y],
		 	[z]
		], dtype=float) / magnitude	


		# compute amount of rotation
		rotation_angle = magnitude * (1000 / sample_rate)

		# compute estimated rotation axis
		orientation_change = get_quaternion(rotation_axis, rotation_angle)
		
		# compute quaternion change
		orientation = quaternion_product(orientation_change, previous_obs)

		# return quaternion difference between current and previously observed orientation
		return orientation
	
	# simple dead reckoning filter with just gyro readings
	if mode == 1:
		orientation = drf(data[" gyroscope.X"], data[" gyroscope.Y"], data[" gyroscope.Z"]) 
		orientation_inv = quaternion_conjugation(orientation)
		return orientation, orientation_inv
	
	# dead reckoning filter with accelerometer readings integrated
	if mode == 2:
		orientation = drf(data[" gyroscope.X"], data[" gyroscope.Y"], data[" gyroscope.Z"]) 
		orientation_inv = quaternion_conjugation(orientation)
		
		# get accel. sensor readings
		a = [0, data[" accelerometer.X"], data[" accelerometer.Y"], data[" accelerometer.Z"]]
		
		# convert readings to world frame and normalise
		world_a = get_world_accel(a, orientation, orientation_inv)	
		world_a = getAxis(world_a[0],world_a[1],world_a[2], world_a[3])
		world_a = world_a /np.sqrt((world_a[0] ** 2) + (world_a[1] ** 2) + (world_a[2] ** 2)) 

		# compute angle between up and accelerometer vectors
		phi = np.arccos(world_a[1])
		
		# fusion of gyroscope and accelerometer signals
		alpha = 0.9
		n = np.array([
			[-world_a[2]], 
			[0],
			 [world_a[0]]
			 ])

		# normalise n
		n_mag = np.sqrt((n[0] ** 2) + (n[1] ** 2) + (n[2] ** 2))
		if n_mag == 0 or n_mag == nan:
			n_norm = n
		else:
			n_norm = n / n_mag

		# construct orientation quaternion
		q_c = quaternion_product(get_quaternion(n_norm, (1-alpha)*phi), orientation)
		q_c_inv = quaternion_conjugation(q_c)
		
		# return orientation and inverse
		return q_c, q_c_inv
	# dead reckoning filter with both accelerometer and magnetometer readings integrated
	else:
		orientation = drf(data[" gyroscope.X"], data[" gyroscope.Y"], data[" gyroscope.Z"]) 
		orientation_inv = quaternion_conjugation(orientation)
		
		# get accel. sensor readings
		a = [0, data[" accelerometer.X"], data[" accelerometer.Y"], data[" accelerometer.Z"]]
		
		# convert readings to world frame and normalise
		world_a = get_world_accel(a, orientation, orientation_inv)
		world_a = getAxis(world_a[0],world_a[1],world_a[2], world_a[3])
		world_a = world_a /np.sqrt((world_a[0] ** 2) + (world_a[1] ** 2) + (world_a[2] ** 2)) 

		# compute angle between up and accelerometer vectors
		phi = np.arccos(world_a[1])
		
		# fusion of gyroscope and accelerometer signals
		alpha = 0.1
		n = np.array([
			[-world_a[2]], 
			[0],
			 [world_a[0]]
			 ])

		# normalise n
		n_mag = np.sqrt((n[0] ** 2) + (n[1] ** 2) + (n[2] ** 2))
		if n_mag == 0 or n_mag == nan:
			n_norm = n
		else:
			n_norm = n / n_mag

		# construct orientation quaternion
		q_c = quaternion_product(get_quaternion(n_norm, (1-alpha)*phi), orientation)
		q_c_inv = quaternion_conjugation(q_c)

		m = [0, data[" magnetometer.X"], data[" magnetometer.Y"], data[" magnetometer.Z "]]

		alpha2 = 0.5
		theta = np.arctan2(m[1], m[3])
		theta_r = np.arctan2(mag_data[1], mag_data[3])

		q_c = quaternion_product(get_quaternion(np.array([[0], [1], [0]]), -alpha2*(theta-theta_r)), orientation)
		q_c_inv = quaternion_conjugation(q_c)

		
		# return orientation and inverse
		return q_c, q_c_inv

# sequence 1
def dead_reckoning_filter(model, df, projection_type="perspective"):
	# array to track the observations at each timestep
	orientation_data = []
	plt.ion()
	for i in range(len(df)):
		# init image
		image = Image(width, height, Color(255, 255, 255, 255))

		# Init z-buffer
		zBuffer = [-float('inf')] * width * height
		
		# get previous observation
		if i == 0:
			orientation = [1,0,0,0]
		else:
			orientation = orientation_data[i-1]

		# get estimated transformation quaternion and inverse from sensor measurements at current timestep
		q_obs, q_inv = complementary_filter(df.iloc[i], orientation, mode=1)

		# save orientation data for next iteration
		orientation_data.append(q_obs)

		for face in model.faces:
			p0, p1, p2=[model.vertices[i] for i in face]
			n0, n1, n2=[vertexNormals[i] for i in face]

			# Define the light direction
			lightDir=Vector(0, 0, -1)

			# Set to true if face should be culled
			cull=False

			# Transform vertices and calculate lighting intensity per vertex
			transformedPoints=[]
			for p, n in zip([p0, p1, p2], [n0, n1, n2]):
				# convert points to quaternion
				p_ = [0, p.x, p.y, p.z]

				# rotation: qpq^-1
				inner = quaternion_product(q_obs,p_)
				q0, q1, q2, q3 = quaternion_product(inner,q_inv)

				# convert quaternion to x, y, z
				px, py, pz = getAxis(q0, q1, q2, q3)

				# get rotation matrix from quaternions for pixel intensity computation 
				M = np.matrix([
					[1-2*q2**2-2*q3**2, 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2],
					[2*q1*q2-2*q0*q3, 1-2*q1**2-2*q3**2 , 2*q2*q3+2*q0*q1],
					[2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 1-2*q1**2-2*q2**2],
				], dtype=float)

				if projection_type == "orthographic":
					# normal vector
					n_ = np.array([
						[n.x],
						[n.y], 
						[n.z], 
					], dtype=float)

					# compute intensities 
					n_ = (np.linalg.inv(V[:3, :3]) * np.linalg.inv(M)).T * n_

					# perform projection
					screenX, screenY = getOrthographicProjection(px, py, pz)

					# compute pixel intensities as a combination of transformed vertices and normals
					vec_intensities = [screenX, screenY, p.z]
					intensity =np.clip( Vector(vec_intensities[0], vec_intensities[1], vec_intensities[2]).normalize() * lightDir, a_min=None, a_max=1.0)

					# Intensity < 0 means light is shining through the back of the face
					# In this case, don't draw the face at all ("back-face culling")
					if intensity < 0:
						cull=True
						break

					transformedPoints.append(Point(screenX, screenY, pz, Color(intensity*255, intensity*255, intensity*255, 255)))

				else:
					screenX, screenY, screenZ, n_ = getPerspectiveProjection(V,M, px, py, pz, n)
					
					# compute pixel intensities as a combination of transformed vertices and normals
					vec_intensities = [screenX+n_[0, 0], screenY+n_[1, 0], screenZ+n_[2, 0]]
					intensity =np.clip( Vector(vec_intensities[0], vec_intensities[1], vec_intensities[2]).normalize() * lightDir, a_min=None, a_max=1.0)

					# Intensity < 0 means light is shining through the back of the face
					# In this case, don't draw the face at all ("back-face culling")
					if intensity < 0:
						cull=True
						break

					transformedPoints.append(Point(screenX, screenY, screenZ, Color(intensity*255, intensity*255, intensity*255, 255)))

			if not cull:
				Triangle(transformedPoints[0], transformedPoints[1],
						transformedPoints[2]).draw(image, zBuffer)
		# real time output 
		img = image.saveAsPNG()
		plt.imshow(img)
		plt.pause(0.000001)

# sequence 2
def accel_filter(model, df, projection_type="perspective"):
	# array to track the observations at each timestep
	orientation_data = []
	plt.ion()
	for i in range(len(df)):
		# init image
		image = Image(width, height, Color(255, 255, 255, 255))

		# Init z-buffer
		zBuffer = [-float('inf')] * width * height

		# get previous observation
		if i == 0:
			orientation = [1,0,0,0]
		else:
			orientation = orientation_data[i-1]

		# get estimated transformation quaternion and inverse from given sensor measurements at current timestep
		q_obs, q_inv = complementary_filter(df.iloc[i], orientation, mode=2)

		# get estimated transformation quaternion and inverse from sensor measurements at current timestep
		orientation_data.append(q_obs)

		for face in model.faces:
			p0, p1, p2=[model.vertices[i] for i in face]
			n0, n1, n2=[vertexNormals[i] for i in face]

			# Define the light direction
			lightDir=Vector(0, 0, -1)

			# Set to true if face should be culled
			cull=False

			# Transform vertices and calculate lighting intensity per vertex
			transformedPoints=[]
			for p, n in zip([p0, p1, p2], [n0, n1, n2]):
				# convert points to quaternion
				p_ = [0, p.x, p.y, p.z]

				# rotation: qpq^-1
				inner = quaternion_product(q_obs,p_)
				q0, q1, q2, q3 = quaternion_product(inner,q_inv)

				# convert quaternion to x, y, z
				px, py, pz = getAxis(q0, q1, q2, q3)

				# get rotation matrix from quaternions for pixel intensity computation 
				M = np.matrix([
					[1-2*q2**2-2*q3**2, 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2],
					[2*q1*q2-2*q0*q3, 1-2*q1**2-2*q3**2 , 2*q2*q3+2*q0*q1],
					[2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 1-2*q1**2-2*q2**2],
				], dtype=float)

				if projection_type == "orthographic":
					# normal vector
					n_ = np.array([
						[n.x],
						[n.y], 
						[n.z], 
					], dtype=float)

					# compute intensities 
					n_ = (np.linalg.inv(V[:3, :3]) * np.linalg.inv(M)).T * n_

					# perform projection
					screenX, screenY = getOrthographicProjection(px, py, pz)

					# compute pixel intensities as a combination of transformed vertices and normals
					vec_intensities = [screenX, screenY, p.z]
					intensity =np.clip( Vector(vec_intensities[0], vec_intensities[1], vec_intensities[2]).normalize() * lightDir, a_min=None, a_max=1.0)

					# Intensity < 0 means light is shining through the back of the face
					# In this case, don't draw the face at all ("back-face culling")
					if intensity < 0:
						cull=True
						break

					transformedPoints.append(Point(screenX, screenY, pz, Color(intensity*255, intensity*255, intensity*255, 255)))

				else:
					screenX, screenY, screenZ, n_ = getPerspectiveProjection(V,M, px, py, pz, n)
					
					# compute pixel intensities as a combination of transformed vertices and normals
					vec_intensities = [screenX+n_[0, 0], screenY+n_[1, 0], screenZ+n_[2, 0]]
					intensity =np.clip( Vector(vec_intensities[0], vec_intensities[1], vec_intensities[2]).normalize() * lightDir, a_min=None, a_max=1.0)

					# Intensity < 0 means light is shining through the back of the face
					# In this case, don't draw the face at all ("back-face culling")
					if intensity < 0:
						cull=True
						break

					transformedPoints.append(Point(screenX, screenY, screenZ, Color(intensity*255, intensity*255, intensity*255, 255)))

			if not cull:
				Triangle(transformedPoints[0], transformedPoints[1],
						transformedPoints[2]).draw(image, zBuffer)
		# real time output 
		img = image.saveAsPNG()
		plt.imshow(img)
		plt.pause(0.000001)

# sequence 3
def mag_filter(model, df, projection_type="projection"):
	# array to track the observations at each timestep
	orientation_data = []
	plt.ion()
	for i in range(len(df)):
		# init image
		image = Image(width, height, Color(255, 255, 255, 255))

		# Init z-buffer
		zBuffer = [-float('inf')] * width * height

		# get previous observation
		if i == 0:
			orientation = [1,0,0,0]
			mag_data = [0, df.iloc[i][" magnetometer.X"], df.iloc[i][" magnetometer.Y"],df.iloc[i][" magnetometer.Z "]]
		else:
			orientation = orientation_data[i-1]

		# get estimated transformation quaternion and inverse from given sensor measurements at current timestep
		q_obs, q_inv = complementary_filter(df.iloc[i], orientation, mode=3, mag_data=mag_data)

		# get estimated transformation quaternion and inverse from sensor measurements at current timestep
		orientation_data.append(q_obs)

		for face in model.faces:
			p0, p1, p2=[model.vertices[i] for i in face]
			n0, n1, n2=[vertexNormals[i] for i in face]

			# Define the light direction
			lightDir=Vector(0, 0, -1)

			# Set to true if face should be culled
			cull=False

			# Transform vertices and calculate lighting intensity per vertex
			transformedPoints=[]
			for p, n in zip([p0, p1, p2], [n0, n1, n2]):
				# convert points to quaternion
				# p_ = [p.x, p.y, p.z, 1]
				p_ = [0, p.x, p.y, p.z]

				# rotation: qpq^-1
				inner = quaternion_product(q_obs,p_)
				q0, q1, q2, q3 = quaternion_product(inner,q_inv)

				# convert quaternion to x, y, z
				px, py, pz = getAxis(q0, q1, q2, q3)

				# get rotation matrix from quaternions for pixel intensity computation 
				M = np.matrix([
					[1-2*q2**2-2*q3**2, 2*q1*q2+2*q0*q3, 2*q1*q3-2*q0*q2],
					[2*q1*q2-2*q0*q3, 1-2*q1**2-2*q3**2 , 2*q2*q3+2*q0*q1],
					[2*q1*q3+2*q0*q2, 2*q2*q3-2*q0*q1, 1-2*q1**2-2*q2**2],
				], dtype=float)

				if projection_type == "orthographic":
					# normal vector
					n_ = np.array([
						[n.x],
						[n.y], 
						[n.z], 
					], dtype=float)

					# compute intensities 
					n_ = (np.linalg.inv(V[:3, :3]) * np.linalg.inv(M)).T * n_

					# perform projection
					screenX, screenY = getOrthographicProjection(px, py, pz)

					# compute pixel intensities as a combination of transformed vertices and normals
					vec_intensities = [screenX, screenY, p.z]
					intensity =np.clip( Vector(vec_intensities[0], vec_intensities[1], vec_intensities[2]).normalize() * lightDir, a_min=None, a_max=1.0)

					# Intensity < 0 means light is shining through the back of the face
					# In this case, don't draw the face at all ("back-face culling")
					if intensity < 0:
						cull=True
						break

					transformedPoints.append(Point(screenX, screenY, pz, Color(intensity*255, intensity*255, intensity*255, 255)))

				else:
					screenX, screenY, screenZ, n_ = getPerspectiveProjection(V,M, px, py, pz, n)
					
					# compute pixel intensities as a combination of transformed vertices and normals
					vec_intensities = [screenX+n_[0, 0], screenY+n_[1, 0], screenZ+n_[2, 0]]

					intensity =np.clip( Vector(vec_intensities[0], vec_intensities[1], vec_intensities[2]).normalize() * lightDir, a_min=None, a_max=1.0)

					# Intensity < 0 means light is shining through the back of the face
					# In this case, don't draw the face at all ("back-face culling")
					if intensity < 0:
						cull=True
						break

					transformedPoints.append(Point(screenX, screenY, screenZ, Color(intensity*255, intensity*255, intensity*255, 255)))

			if not cull:
				Triangle(transformedPoints[0], transformedPoints[1],
						transformedPoints[2]).draw(image, zBuffer)
		# real time output 
		img = image.saveAsPNG()
		plt.imshow(img)
		plt.pause(0.000001)

# Initialisation and constant definitions
width = 500
height = 300

# get filter mode
mode = int(sys.argv[3])

# get model path
model_path = sys.argv[2]

# Load the model
model = Model(model_path)
model.normalizeGeometry()

# get dataset path
data_path = sys.argv[1]

# load dataframe and perform conversions necessary normalisations
data = pd.read_csv(data_path)
df=convert_data(data)

# Calculate face normals
faceNormals={}
for face in model.faces:
	p0, p1, p2=[model.vertices[i] for i in face]
	faceNormal=(p2-p0).cross(p1-p0).normalize()

	for i in face:
		if not i in faceNormals:
			faceNormals[i]=[]

		faceNormals[i].append(faceNormal)

# Calculate vertex normals
vertexNormals=[]
for vertIndex in range(len(model.vertices)):
	vertNorm=getVertexNormal(vertIndex, faceNormals)
	vertexNormals.append(vertNorm)

# get view transform matrix
V = getViewTransform()

if mode == 1:
	# Dead-reckoning filter
	print("Sequence 1: Simple dead-reckoning filter...")
	dead_reckoning_filter(model, df)
elif mode == 2:
	# Dead-reckoning filter with accelerometer
	print("Sequence 2: Dead-reckoning filter with accelerometer integration...")
	accel_filter(model, df)
else:
	# Dead-reckoning filter with accelerometer and magnetometer
	print("Sequence 3: Dead-reckoning filter with accelerometer and magnetometer integration...")
	mag_filter(model, df)