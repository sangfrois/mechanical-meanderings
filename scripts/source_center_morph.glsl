// Source-contour emergence confined to StreamDiffusion's fixed center crop.
// Input 0 is the original source; input 1 is its temporally filtered diffusion.
uniform vec4 uMorph; // amount, warp, softness, time
out vec4 fragColor;
float lum(vec3 c){ return dot(c,vec3(.2126,.7152,.0722)); }
float field(vec2 uv){
 vec2 px=uTD2DInfos[0].res.xy*8.0;
 return (lum(texture(sTD2DInputs[0],uv).rgb)*2.0+
 lum(texture(sTD2DInputs[0],uv+vec2(px.x,0)).rgb)+
 lum(texture(sTD2DInputs[0],uv-vec2(px.x,0)).rgb)+
 lum(texture(sTD2DInputs[0],uv+vec2(0,px.y)).rgb)+
 lum(texture(sTD2DInputs[0],uv-vec2(0,px.y)).rgb))/6.0;
}
void main(){
 vec2 uv=vUV.st;
 float a=clamp(uMorph.x,0.0,1.0);
 float pulse=4.0*a*(1.0-a);
 vec2 px=uTD2DInfos[0].res.xy*5.0;
 float f=field(uv);
 vec2 grad=vec2(field(uv+vec2(px.x,0))-field(uv-vec2(px.x,0)),field(uv+vec2(0,px.y))-field(uv-vec2(0,px.y)));
 vec2 flow=grad/(0.15+length(grad));
 flow.x*=uTDOutputInfo.res.w/uTDOutputInfo.res.z;
 float soft=max(.01,uMorph.z);
 float threshold=mix(-soft,1.0+soft,a);
 float mask=1.0-smoothstep(threshold-soft,threshold+soft,clamp(f,.001,.999));
 float srcAspect=uTD2DInfos[0].res.z/uTD2DInfos[0].res.w;
 float genAspect=uTD2DInfos[1].res.z/uTD2DInfos[1].res.w;
 vec2 cropScale=vec2(max(1.0,srcAspect/genAspect),max(1.0,genAspect/srcAspect));
 vec2 cropUV=(uv-0.5)*cropScale+0.5;
 // The mask is evaluated BEFORE displacement, so the crop cannot move.
 vec2 edge=min(cropUV,1.0-cropUV);
 float boundary=smoothstep(0.0,0.15,min(edge.x,edge.y));
 mask*=boundary;
 vec2 gu=clamp(cropUV+flow*uMorph.y*pulse*(1.0-a)*boundary,0.0,1.0);
 // Never warp source coordinates, including inside the generated region.
 vec4 src=texture(sTD2DInputs[0],uv);
 vec4 gen=texture(sTD2DInputs[1],gu);
 fragColor=TDOutputSwizzle(mix(src,gen,mask));
}
