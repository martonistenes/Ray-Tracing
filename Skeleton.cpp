//=============================================================================================
// Masodik hazi feladat: Orbifold vizualizacio
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : Istenes Marton
// Neptun : ASDSVN
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#include "framework.h"

const char* const vertexSource = R"(
	#version 330
	precision highp float;

	uniform vec3 wLookAt, wRight, wUp;
	layout(location = 0) in vec2 cCamWindowVertex;
	out vec3 p;

	void main() {
		gl_Position = vec4(cCamWindowVertex, 0, 1);
		p = wLookAt + wRight * cCamWindowVertex.x + wUp * cCamWindowVertex.y;
	}
)";

const char* const fragmentSource = R"(
	#version 330
	precision highp float;

	in vec3 p;
	out vec4 fragmentColor;

	const vec3 La = vec3(0.53, 0.81, 0.92);
	const vec3 ka = vec3(0.5f, 0.5f, 0.5f);

	const vec3 Le = vec3(1, 1, 1);
	const vec3 lightPosition = vec3(0.0f, 0.0f, -1.0f);
	const float shininess = 500.0f;

	const int maxdepth = 6;
	const float epsilon = 0.01f;

	uniform vec3 wEye, v[20];
	uniform int planes[60];
	uniform vec3 kd, ks, F_Gold, F_Mirror;

	struct Hit {
		float t;
		vec3 position, normal;
		int mat;
	};

	struct Ray {
		vec3 start, dir, weight;
	};

	mat4 RotationMatrix(float angle, vec3 w) {
		float c = cos(angle), s = sin(angle);
		w = normalize(w);

		return mat4(vec4(c * (1 - w.x*w.x) + w.x*w.x, w.x*w.y*(1 - c) + w.z*s, w.x*w.z*(1 - c) - w.y*s, 0),
					vec4(w.x*w.y*(1 - c) - w.z*s, c * (1 - w.y*w.y) + w.y*w.y, w.y*w.z*(1 - c) + w.x*s, 0),
					vec4(w.x*w.z*(1 - c) + w.y*s, w.y*w.z*(1 - c) - w.x*s, c * (1 - w.z*w.z) + w.z*w.z, 0),
					vec4(0, 0, 0, 1));
	}

	vec3 rotateWith72(vec3 point, vec3 axis ){
		vec4 temp1 = vec4(point.x,point.y,point.z, 1);
		vec4 temp2 = temp1 * RotationMatrix(0.4 * 3.141592653, axis);
		return vec3(temp2.x,temp2.y,temp2.z);
	}

	void getDodecaPlane(int i, out vec3 p, out vec3 normal, out vec3 polygon[5]) {

		vec3[5] planevertex;

		for(int k = 0; k < 5; k++){
			planevertex[k] = v[planes[5 * i + k] - 1];
		}
		

		normal = cross(planevertex[1] - planevertex[0], planevertex[2] - planevertex[0]);
		if (dot(planevertex[0], normal) < 0) normal = - normal;
		p = planevertex[0] + vec3(0, 0, 0.03f);

		vec3 planeorigo = vec3(0, 0, 0);
		for(int k = 0; k < 5; k++){
			planeorigo = planeorigo + planevertex[k] / 5;
		}

		for(int k = 0; k < 5; k++){
			polygon[k] = planevertex[k] + normalize(planeorigo - planevertex[k]) * 0.1f;
		}

	}

	Hit intersectDodeca(Ray ray, Hit hit) {

		for(int i = 0; i < 12; i++) {
			vec3 p1, normal;
			vec3 planevertex[5];
			getDodecaPlane(i, p1, normal, planevertex);

			float ti = abs(dot(normal, ray.dir)) > epsilon ? dot(p1 - ray.start, normal) / dot(normal, ray.dir) : -1;
			if (ti <= epsilon || (ti > hit.t && hit.t > 0)) continue;
			vec3 pintersect = ray.start + ray.dir * ti;
			bool outside = false;
			for(int j = 0; j < 12; j++)  {
				if (i == j) continue;
				vec3 p11, n;
				vec3 temp[5];
				getDodecaPlane(j, p11, n, temp);
				if(dot(n, pintersect - p11) > 0) {
					outside = true;
					break;
				}
			}
			if(!outside) {

				hit.t = ti;
				hit.position = pintersect;
				hit.normal = normalize(normal);
				
				bool b1 = dot(cross(planevertex[1] - planevertex[0], pintersect - planevertex[0]), normal) > 0;
				bool b2 = dot(cross(planevertex[2] - planevertex[1], pintersect - planevertex[1]), normal) > 0;
				bool b3 = dot(cross(planevertex[3] - planevertex[2], pintersect - planevertex[2]), normal) > 0;
				bool b4 = dot(cross(planevertex[4] - planevertex[3], pintersect - planevertex[3]), normal) > 0;
				bool b5 = dot(cross(planevertex[0] - planevertex[4], pintersect - planevertex[4]), normal) > 0;
				
				if(b1 && b2 && b3 && b4 && b5){hit.mat = 3;}
				else{hit.mat = 1;}
			}
		}
		return hit;
	}

	Hit intersectImplicit(Ray ray, Hit hit){

		const float a = 1;
		const float b = 1;
		const float c = 1;

		float a1 =  a * ray.dir.x * ray.dir.x +  b * ray.dir.y * ray.dir.y;
		float b1 = 2 * (a * ray.start.x * ray.dir.x + b * ray.start.y * ray.dir.y) - c * ray.dir.z;
		float c1 = ( a * ray.start.x * ray.start.x +  b * ray.start.y * ray.start.y) - c * ray.start.z;

		float discr = b1 * b1 -4.0f * a1 * c1;
		if(discr >= 0){
			float sqrt_discr = sqrt(discr);
			float t1 = (-b1 + sqrt_discr) / 2.0f / a1;
			vec3 p  = ray.start + ray.dir * t1;
			if(length(-p) > 0.3) t1 = -1;
			float t2 = (-b1 - sqrt_discr) / 2.0f / a1;
			p = ray.start + ray.dir * t2;
            if(length(-p) > 0.3) t2 = -1;
			if (t2 > 0 && (t2 < t1 || t1 < 0)) t1 = t2;
			if (t1 > 0 && (t1 < hit.t || hit.t < 0)){

				hit.t = t1;
				hit.position = ray.start + ray.dir * hit.t;
				hit.normal = normalize(vec3(2 * a * hit.position.x, 2 * b * hit.position.y, -c));
				hit.mat = 2;

			}
		}

		return hit;
	}

	Hit firstIntersect(Ray ray) {
		Hit bestHit;
		bestHit.t = -1;
		bestHit = intersectImplicit(ray, bestHit);
		bestHit = intersectDodeca(ray, bestHit);
		if (dot(ray.dir, bestHit.normal) > 0) bestHit.normal = bestHit.normal * (-1);
		return bestHit;
	}
	
	vec3 trace(Ray ray) {
		vec3 outRadiance = vec3(0, 0, 0);
		for(int d = 0; d < maxdepth; d++) {
			Hit hit = firstIntersect(ray);

			if (hit.t < 0) break;

			if (hit.mat == 1) {
				vec3 lightdir = normalize(lightPosition - hit.position);
				float cosTheta = dot(hit.normal, lightdir);
				if(cosTheta > 0) {
					vec3 LeIn = Le / dot(lightPosition - hit.position, lightPosition - hit.position);
					outRadiance += ray.weight * LeIn * kd * cosTheta;
					vec3 halfway = normalize(-ray.dir + lightdir);
					float cosDelta = dot(hit.normal, halfway);
					if (cosDelta > 0) outRadiance += ray.weight * LeIn * ks * pow(cosDelta, shininess);
				}
				ray.weight *= ka;
				break;
			}

			if(hit.mat == 2) {
				ray.weight *= F_Gold + (vec3(1,1,1) - F_Gold) * pow(dot(-ray.dir, hit.normal), 5);
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
			}

			if(hit.mat == 3) {
				ray.weight *= F_Mirror + (vec3(1,1,1) - F_Mirror) * pow(dot(-ray.dir, hit.normal), 5);
				ray.start = hit.position + hit.normal * epsilon;
				ray.dir = reflect(ray.dir, hit.normal);
				ray.start = rotateWith72(ray.start,hit.normal);
				ray.dir = rotateWith72(ray.dir,hit.normal);
			}
		}
		outRadiance += ray.weight * La;
		return outRadiance;
	}

	void main() {
		Ray ray;
		ray.start = wEye; 
		ray.dir = normalize(p - wEye);
		ray.weight = vec3(1,1,1);
		fragmentColor = vec4(trace(ray), 1); 
	}
)";

struct Camera {

	vec3 eye, lookat, right, pvup, rvup;
	float fov = 45 * (float)M_PI / 180;

	Camera() : eye(0, 1, 1), pvup(0, 0, 1), lookat(0, 0, 0) { set(); }

	void set() {

		vec3 w = eye - lookat;
		float f = length(w);
		right = normalize(cross(pvup, w)) * f * tanf(fov / 2);
		rvup = normalize(cross(w, right)) * f * tanf(fov / 2);
	}
	void Animate(float t) {
		float r = sqrtf(eye.x * eye.x + eye.y * eye.y);
		eye = vec3(r * cos(t) + lookat.x, r * sin(t) + lookat.y, eye.z);
		set();
	}

};

GPUProgram shader;
Camera camera;

float F(float n, float k) { return ((n - 1) * (n - 1) + k * k) / ((n + 1) * (n + 1) + k * k); }

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	unsigned int vao, vbo;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);

	float vertexCoords[] = { -1, -1, 1, -1, 1, 1, -1, 1 };
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertexCoords), vertexCoords, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);

	shader.create(vertexSource, fragmentSource, "fragmentColor");

	const float g = 0.618f, G = 1.618f;
	std::vector<vec3> v = {
		vec3(0,g,G),vec3(0,-g,G),vec3(0,-g,-G),vec3(0,g,-G),
		vec3(G,0,g),vec3(-G,0,g),vec3(-G,0,-g),vec3(G,0,-g),
		vec3(g,G,0),vec3(-g,G,0),vec3(-g,-G,0),vec3(g,-G,0),
		vec3(1,1,1),vec3(-1,1,1),vec3(-1,-1,1),vec3(1,-1,1),
		vec3(1,-1,-1),vec3(1,1,-1),vec3(-1,1,-1),vec3(-1,-1,-1),
	};
	for (int i = 0; i < v.size(); i++)
	{
		shader.setUniform(v[i], "v[" + std::to_string(i) + "]");
	}

	int planes[60] = {
		1,2,16,5,13, 1,13,9,10,14, 1,14,6,15,2, 2,15,11,12,16, 3,4,18,8,17, 3,17,12,11,20, 3,20,7,19,4, 19,10,9,18,4, 16,12,17,8,5, 5,8,18,9,13, 14,10,19,7,6, 6,7,20,11,15
	};

	for (int i = 0; i < 60; i++)
	{
		shader.setUniform(planes[i], "planes[" + std::to_string(i) + "]");
	}

	shader.setUniform(vec3(0.85, 0.64, 0.12), "kd");
	shader.setUniform(vec3(5, 5, 5), "ks");
	shader.setUniform(vec3(F(0.17, 3.1), F(0.35, 2.7), F(2.5, 1.9)), "F_Gold");
	shader.setUniform(vec3(F(0, 0), F(0, 0), F(0, 0)), "F_Mirror");
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	shader.setUniform(camera.eye, "wEye");
	shader.setUniform(camera.lookat, "wLookAt");
	shader.setUniform(camera.right, "wRight");
	shader.setUniform(camera.rvup, "wUp");

	glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {}

void onIdle() {
	camera.Animate(glutGet(GLUT_ELAPSED_TIME) / 5000.0f);
	glutPostRedisplay();
}
