varying highp vec2 textureCoordinate;

uniform sampler2D inputImageTexture;

// parameters

// RGB <-> HSV, Reference to GPUImageSkinToneFilter.m

 highp vec3 rgb2hsv(highp vec3 c)
{
    highp vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    highp vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    highp vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    
    highp float d = q.x - min(q.w, q.y);
    highp float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// HSV <-> RGB, Reference to GPUImageSkinToneFilter.m
 highp vec3 hsv2rgb(highp vec3 c)
{
    highp vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    highp vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main(){
    highp vec4 colorRGB = texture2D(inputImageTexture, textureCoordinate);

    // Convert color to HSV, extract hue
    highp vec3 colorHSV = rgb2hsv(colorRGB.rgb);

    //highp float Huevariance = 1.0 / 6 / 3;
    //highp float HueUpper = 0.11111111;
    //highp float HueLower = 0.22222222;

    if(colorHSV.x < 0.18 && colorHSV.x > 0.10 && colorHSV.y > 0.4 && colorHSV.z > 0.37){
        // make it become white in RGB
        colorHSV.x = 0.0;
        colorHSV.y = 0.0;
        colorHSV.z = 1.0;
        
    }
    else{
        // make it become Black in RGB
        colorHSV.x = 0.0;
        colorHSV.y = 0.0;
        colorHSV.z = 0.0;
    }

    // final color
    highp vec3 finalColorRGB = hsv2rgb(colorHSV.rgb);

    // display
    gl_FragColor = vec4(finalColorRGB, 1.0);
}

/*
void main()
{
    lowp vec3 tc = vec3(1.0, 0.0, 0.0);

    lowp vec3 pixcol = texture2D(inputImageTexture, textureCoordinate).rgb;
    lowp vec3 colors[3];
    colors[0] = vec3(0.0, 0.0, 1.0);
    colors[1] = vec3(1.0, 1.0, 0.0);
    colors[2] = vec3(1.0, 0.0, 0.0);
    mediump float lum = (pixcol.r + pixcol.g + pixcol.b) / 3.0;
    int ix = (lum < 0.5)? 0:1;
    tc = mix(colors[ix], colors[ix + 1], (lum - float(ix) * 0.5) / 0.5);

    gl_FragColor = vec4(tc, 1.0);
}


void main(){
    lowp vec4 textureColor = texture2D(inputImageTexture, textureCoordinate);
    lowp vec4 outputColor;
    outputColor.r = (textureColor.r * 0.393) + (textureColor.g * 0.769) + (textureColor.b * 0.189);
    outputColor.g = (textureColor.r * 0.349) + (textureColor.g * 0.686) + (textureColor.b * 0.168);    
    outputColor.b = (textureColor.r * 0.272) + (textureColor.g * 0.534) + (textureColor.b * 0.131);
    outputColor.a = 1.0;

    gl_FragColor = outputColor;
}
*/
