/*************************************************** 
  This is an example for our Adafruit 16-channel PWM & Servo driver
  Servo test - this will drive 16 servos, one after the other

  Pick one up today in the adafruit shop!
  ------> http://www.adafruit.com/products/815

  These displays use I2C to communicate, 2 pins are required to  
  interface. For Arduino UNOs, thats SCL -> Analog 5, SDA -> Analog 4

  Adafruit invests time and resources providing this open source code, 
  please support Adafruit and open-source hardware by purchasing 
  products from Adafruit!

  Written by Limor Fried/Ladyada for Adafruit Industries.  
  BSD license, all text above must be included in any redistribution
 ****************************************************/

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// called this way, it uses the default address 0x40
Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();
// you can also call it with a different address you want
//Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver(0x41);

// Depending on your servo make, the pulse width min and max may vary, you 
// want these to be as small/large as possible without hitting the hard stop
// for max range. You'll have to tweak them as necessary to match the servos you
// have!
#define SERVOMIN  180  // this is the 'minimum' pulse length count (out of 4096)
#define SERVOMAX  440 // this is the 'maximum' pulse length count (out of 4096)

// our servo # counter
uint8_t servonum = 0;

int Servo1_1=400;
int Servo2_1=415;


int Servo1_2=300;
int Servo2_2=425;


int Servo1_3=300;
int Servo2_3=300;

int Servo1_4=500;
int Servo2_4=400;


int Servo1_5=420;
int Servo2_5=530;

void setup() {
  Serial.begin(9600);
  Serial.println("16 channel Servo test!");

  pwm.begin();
  
  pwm.setPWMFreq(60);  // Analog servos run at ~60 Hz updates
}

// you can use this function if you'd like to set the pulse length in seconds
// e.g. setServoPulse(0, 0.001) is a ~1 millisecond pulse width. its not precise!
void setServoPulse(uint8_t n, double pulse) {
  double pulselength;
  
  pulselength = 1000000;   // 1,000,000 us per second
  pulselength /= 60;   // 60 Hz
  Serial.print(pulselength); Serial.println(" us per period"); 
  pulselength /= 4096;  // 12 bits of resolution
  Serial.print(pulselength); Serial.println(" us per bit"); 
  pulse *= 1000;
  pulse /= pulselength;
  Serial.println(pulse);
  pwm.setPWM(n, 0, pulse);
}

void loop() {
  
pwm.setPWM(2,0,0);
  pwm.setPWM(3,0,0);
 
 pwm.setPWM(0, 0, Servo1_1);
 pwm.setPWM(1,0,Servo2_1);


 delay(2000);

 pwm.setPWM(2,0,4095);
  

  pwm.setPWM(0, 0, Servo1_2);
 pwm.setPWM(1,0,Servo2_2);

 delay(200);
 

  pwm.setPWM(0, 0, Servo1_3);
 pwm.setPWM(1,0,Servo2_3);
pwm.setPWM(3,0,4095);

 delay(200);

  pwm.setPWM(0, 0, Servo1_4);
 pwm.setPWM(1,0,Servo2_4);


 delay(200);

   pwm.setPWM(0, 0, Servo1_5);
 pwm.setPWM(1,0,Servo2_5);
pwm.setPWM(3,0,0);
 

 delay(200);

  pwm.setPWM(2,0,4095);
  

  pwm.setPWM(0, 0, Servo1_2);
 pwm.setPWM(1,0,Servo2_2);

 delay(200);
 

  pwm.setPWM(0, 0, Servo1_3);
 pwm.setPWM(1,0,Servo2_3);
pwm.setPWM(3,0,4095);

 delay(200);

  pwm.setPWM(0, 0, Servo1_4);
 pwm.setPWM(1,0,Servo2_4);


 delay(200);

   pwm.setPWM(0, 0, Servo1_5);
 pwm.setPWM(1,0,Servo2_5);
pwm.setPWM(3,0,0);
 

 delay(200);
 
}
