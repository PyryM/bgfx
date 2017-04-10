/*
 * Copyright 2011-2017 Branimir Karadzic. All rights reserved.
 *                2017 Pyry Matikainen
 * License: https://github.com/bkaradzic/bgfx#license-bsd-2-clause
 */

#include "common.h"
#include "bgfx_utils.h"
#include "imgui/imgui.h"

#define MAX_LIGHTS_PER_TILE 1024
#define TILE_SIZE 16

struct LightBufferVertex
{
	float m_x;
	float m_y;
	float m_z;
	float m_w;

	static void init()
	{
		ms_decl
			.begin()
			.add(bgfx::Attrib::Position, 4, bgfx::AttribType::Float)
			.end();
	}

	static bgfx::VertexDecl ms_decl;
};

bgfx::VertexDecl LightBufferVertex::ms_decl;

struct LightIndexVertex
{
	float m_x;

	static void init()
	{
		ms_decl
			.begin()
			.add(bgfx::Attrib::Position, 1, bgfx::AttribType::Float)
			.end();
	}

	static bgfx::VertexDecl ms_decl;
};

bgfx::VertexDecl LightIndexVertex::ms_decl;

struct LightData
{
	LightBufferVertex*   m_vertices;
	uint32_t             m_vertexCount; // 3 * lightCount
	uint32_t             m_lightCount;

	bgfx::DynamicVertexBufferHandle m_lightBuffer;
	bgfx::DynamicVertexBufferHandle m_visibleLightBuffer;
};


class ExampleFPlus : public entry::AppI
{
	float randf()
	{
		return static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
	}

	void setRandomColor(LightBufferVertex& light) {
		light.m_x = randf() * 0.001f;
		light.m_y = randf() * 0.001f;
		light.m_z = randf() * 0.001f;
		light.m_w = 1.0f;
	}

	void setRandomPos(LightBufferVertex& light) {
		light.m_x = randf() * 2.0f - 1.0f;
		light.m_y = randf() * 2.0f;
		light.m_z = randf() * 2.0f - 1.0f;
		light.m_w = 1.0f;
	}

	void updateLightPosition(LightBufferVertex& light) {
		// lights float down to 0.0 then pop back up to 2.0
		light.m_y -= 0.005f;
		if (light.m_y < 0.0) {
			light.m_y = randf() * 2.0f;
		}
	}

	void initLights() {
		unsigned int idx = 0;
		for (unsigned int i = 0; i < m_lightData.m_lightCount; ++i) {
			// using an unstructured buffer of vec4s: each light needs 3 vals
			setRandomColor(m_lightData.m_vertices[idx + 0]); // color
			setRandomPos(m_lightData.m_vertices[idx + 1]);   // pos
			m_lightData.m_vertices[idx + 2].m_x = 0.2f;      // radius
			idx += 3;
		}

		const bgfx::Memory* mem = bgfx::makeRef(&m_lightData.m_vertices[0], sizeof(LightBufferVertex) * m_lightData.m_vertexCount);
		bgfx::updateDynamicVertexBuffer(m_lightData.m_lightBuffer, 0, mem);
	}

	void updateLights()
	{
		unsigned int pos = 0;
		for (unsigned int i = 0; i < m_lightData.m_lightCount; ++i) {
			updateLightPosition(m_lightData.m_vertices[pos + 1]); // pos
			pos += 3;
		}

		const bgfx::Memory* mem = bgfx::makeRef(&m_lightData.m_vertices[0], sizeof(LightBufferVertex) * m_lightData.m_vertexCount);
		bgfx::updateDynamicVertexBuffer(m_lightData.m_lightBuffer, 0, mem);
	}

	void init(int _argc, char** _argv) BX_OVERRIDE
	{
		Args args(_argc, _argv);

		m_width  = 1280;
		m_height = 720;
		m_debug  = BGFX_DEBUG_TEXT;
		m_reset  = BGFX_RESET_VSYNC;

		m_tiles_x = m_width / TILE_SIZE;
		m_tiles_y = m_height / TILE_SIZE;

		m_showLightCounts = true;
		m_scrollArea = 0;

		bgfx::init(args.m_type, args.m_pciId);
		bgfx::reset(m_width, m_height, m_reset);

		// Enable debug text.
		bgfx::setDebug(m_debug);

		// Set view 0 clear state.
		bgfx::setViewClear(0
				, BGFX_CLEAR_COLOR|BGFX_CLEAR_DEPTH
				, 0xffffffff
				, 1.0f
				, 0
				);
		bgfx::setViewRect(0, 0, 0, bgfx::BackbufferRatio::Equal);

		// Set view 1 clear state to the same.
		bgfx::setViewClear(1
			, BGFX_CLEAR_COLOR | BGFX_CLEAR_DEPTH
			, 0x303030ff
			, 1.0f
			, 0
		);
		bgfx::setViewRect(1, 0, 0, bgfx::BackbufferRatio::Equal);

		// Create depth framebuffer and bind to view 0
		m_fbtextures[0] = bgfx::createTexture2D(uint16_t(m_width), uint16_t(m_height), false, 1, bgfx::TextureFormat::RGBA32F, BGFX_TEXTURE_COMPUTE_WRITE | BGFX_TEXTURE_RT | BGFX_TEXTURE_U_CLAMP | BGFX_TEXTURE_V_CLAMP);
		m_fbtextures[1] = bgfx::createTexture2D(uint16_t(m_width), uint16_t(m_height), false, 1, bgfx::TextureFormat::D24, BGFX_TEXTURE_RT_WRITE_ONLY);
		m_fbh = bgfx::createFrameBuffer(BX_COUNTOF(m_fbtextures), m_fbtextures, true);
		bgfx::setViewFrameBuffer(0, m_fbh);

		// Create vertex stream declaration.
		LightBufferVertex::init();
		LightIndexVertex::init();

		m_lightData.m_lightCount = 1024;
		m_lightData.m_vertexCount = m_lightData.m_lightCount * 3;
		m_lightData.m_vertices = (LightBufferVertex*)BX_ALLOC(entry::getAllocator(), m_lightData.m_vertexCount * sizeof(LightBufferVertex));
		m_lightData.m_lightBuffer = bgfx::createDynamicVertexBuffer(m_lightData.m_vertexCount, LightBufferVertex::ms_decl, BGFX_BUFFER_COMPUTE_READ);
		m_lightData.m_visibleLightBuffer = bgfx::createDynamicVertexBuffer(m_tiles_x * m_tiles_y * MAX_LIGHTS_PER_TILE, LightIndexVertex::ms_decl, 
																			BGFX_BUFFER_COMPUTE_READ_WRITE | BGFX_BUFFER_COMPUTE_FORMAT_32x1 | BGFX_BUFFER_COMPUTE_TYPE_FLOAT);
		initLights();
		//updatelights();

		//     u_screenSize: vec2f (in pixels)
		//     u_lightCount: vec4 (x: number of lights)
		//     u_projectionMat: camera projection matrix
		//     u_viewMat: camera view matrix
		//     u_dispatchParams: vec4f (x: n_tiles_x, y: n_tiles_y)
		u_screenSize = bgfx::createUniform("u_screenSize", bgfx::UniformType::Vec4);
		u_lightCount = bgfx::createUniform("u_lightCount", bgfx::UniformType::Vec4);
		u_dispatchParams = bgfx::createUniform("u_dispatchParams", bgfx::UniformType::Vec4);
		u_viewMat = bgfx::createUniform("u_viewMat", bgfx::UniformType::Mat4);
		u_projectionMat = bgfx::createUniform("u_projectionMat", bgfx::UniformType::Mat4);
		u_projectionInvMat = bgfx::createUniform("u_projectionInvMat", bgfx::UniformType::Mat4);
		s_depthMap = bgfx::createUniform("s_depthMap", bgfx::UniformType::Int1);

		u_diffuseColor = bgfx::createUniform("u_diffuseColor", bgfx::UniformType::Vec4);
		u_ambientColor = bgfx::createUniform("u_ambientColor", bgfx::UniformType::Vec4);

		// Create program from shaders.
		m_program_depthpass = loadProgram("vs_tiled_lighting_depth", "fs_tiled_lighting_depth");
		m_program_compute = bgfx::createProgram(loadShader("cs_tiled_lighting_cull"), true);
		m_program_light = loadProgram("vs_tiled_lighting_accumulation", "fs_tiled_lighting_accumulation");
		m_program_light_dbg = loadProgram("vs_tiled_lighting_accumulation", "fs_tiled_lighting_debug");

		m_mesh = meshLoad("meshes/bunny.bin");

		m_timeOffset = bx::getHPCounter();

		// Imgui.
		imguiCreate();
	}

	int shutdown() BX_OVERRIDE
	{
		imguiDestroy();
		meshUnload(m_mesh);

		// Cleanup.
		// TODO

		// Shutdown bgfx.
		bgfx::shutdown();

		return 0;
	}

	bool update() BX_OVERRIDE
	{
		if (!entry::processEvents(m_width, m_height, m_debug, m_reset, &m_mouseState) )
		{
			// Set both views to use full viewports.
			bgfx::setViewRect(0, 0, 0, uint16_t(m_width), uint16_t(m_height) );
			bgfx::setViewRect(1, 0, 0, uint16_t(m_width), uint16_t(m_height));
			bgfx::setViewFrameBuffer(0, m_fbh); // WEIRD: just setting this once doesn't work?

			// This dummy draw call is here to make sure that view 0 is cleared
			// if no other draw calls are submitted to view 0.
			//bgfx::touch(1);

			int64_t now = bx::getHPCounter();
			static int64_t last = now;
			const int64_t frameTime = now - last;
			last = now;
			const double freq = double(bx::getHPFrequency() );
			const double toMs = 1000.0/freq;
			float time = (float)( (bx::getHPCounter()-m_timeOffset)/double(bx::getHPFrequency() ) );

			// Use debug font to print information about this example.
			bgfx::dbgTextClear();
			bgfx::dbgTextPrintf(0, 1, 0x4f, "bgfx/examples/34-tiled-lighting");
			bgfx::dbgTextPrintf(0, 2, 0x6f, "Description: Tiled forward (forward plus) lighting.");
			bgfx::dbgTextPrintf(0, 3, 0x0f, "Frame: % 7.3f[ms]", double(frameTime)*toMs);

			// Imgui stuff
			imguiBeginFrame(m_mouseState.m_mx
				, m_mouseState.m_my
				, (m_mouseState.m_buttons[entry::MouseButton::Left] ? IMGUI_MBUT_LEFT : 0)
				| (m_mouseState.m_buttons[entry::MouseButton::Right] ? IMGUI_MBUT_RIGHT : 0)
				| (m_mouseState.m_buttons[entry::MouseButton::Middle] ? IMGUI_MBUT_MIDDLE : 0)
				, m_mouseState.m_mz
				, uint16_t(m_width)
				, uint16_t(m_height)
			);

			imguiBeginScrollArea("Settings", m_width - m_width / 5 - 10, 10, m_width / 5, m_height / 8, &m_scrollArea);

			if (imguiCheck("Show lighting", !m_showLightCounts))
			{
				m_showLightCounts = false;
			}
			if (imguiCheck("Show light counts", m_showLightCounts))
			{
				m_showLightCounts = true;
			}

			imguiEndScrollArea();
			imguiEndFrame();

			// UPDATE LIGHTS
			updateLights();

			// MATRIX SETUP
			float at[3] = { 0.0f, 1.0f,  0.0f };
			float eye[3] = { 0.0f, 1.5f, -2.0f };
			float view[16];
			bx::mtxLookAt(view, eye, at);

			float proj[16];
			float projInv[16];
			bx::mtxProj(proj, 60.0f, float(m_width) / float(m_height), 0.1f, 100.0f, bgfx::getCaps()->homogeneousDepth);
			bx::mtxInverse(projInv, proj);
			bgfx::setViewTransform(0, view, proj);
			bgfx::setViewTransform(1, view, proj);

			// LIGHT CULLING COMPUTE SHADER
			// INPUTS/SETUP:
			//   buffers:
			//     0: lightBuffer [vec4f color, vec4f pos, vec4f rad]*nlights
			//     1: outLightIndices [float]*(MAX_LIGHTS_PER_TILE*NUM_TILES)
			//     2: s_depthMap [rgba32f "depth" buffer]
			//
			//  uniforms:
			//     u_screenSize: vec2f (in pixels)
			//     u_lightCount: vec4 (x: number of lights)
			//     u_projectionMat: camera projection matrix
			//     u_viewMat: camera view matrix
			//     u_dispatchParams: vec4f (x: n_tiles_x, y: n_tiles_y)
			//
			//  Dispatch:
			//     workGroupsX = (SCREEN_SIZE.x + (SCREEN_SIZE.x % 16)) / 16;
			//     workGroupsY = (SCREEN_SIZE.y + (SCREEN_SIZE.y % 16)) / 16;
			//     (workGroupsX, workGroupsY, 1)

			bgfx::setBuffer(0, m_lightData.m_lightBuffer, bgfx::Access::Read);
			bgfx::setBuffer(1, m_lightData.m_visibleLightBuffer, bgfx::Access::ReadWrite);
			bgfx::setImage(2, s_depthMap, m_fbtextures[0], 0, bgfx::Access::Read, bgfx::TextureFormat::RGBA32F);

			float screensize[4] = { float(m_width), float(m_height), 0.0f, 0.0f };
			bgfx::setUniform(u_screenSize, screensize);
			float lightcount[4] = { float(m_lightData.m_lightCount), 0.0f, 0.0f, 0.0f };
			bgfx::setUniform(u_lightCount, lightcount);
			bgfx::setUniform(u_projectionMat, proj);
			bgfx::setUniform(u_projectionInvMat, projInv);
			bgfx::setUniform(u_viewMat, view);
			float dispatchparams[4] = { float(m_tiles_x), float(m_tiles_y), 1.0f, 0.0f };
			bgfx::setUniform(u_dispatchParams, dispatchparams);

			bgfx::dispatch(1, m_program_compute, uint16_t(m_tiles_x), uint16_t(m_tiles_y), 1);

			// DRAW GEOMETRY (both view 0 [depth] and view 1 [light accumulation])
			float mtx[16];
			bx::mtxRotateXY(mtx
				, 0.0f
				, time*0.1f
				);

			meshSubmit(m_mesh, 0, m_program_depthpass, mtx);

			bgfx::setBuffer(0, m_lightData.m_lightBuffer, bgfx::Access::Read);
			bgfx::setBuffer(1, m_lightData.m_visibleLightBuffer, bgfx::Access::Read);
			bgfx::setUniform(u_dispatchParams, dispatchparams);
			float tempcolor1[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
			float tempcolor2[4] = { 0.1f, 0.1f, 0.1f, 0.0f };
			bgfx::setUniform(u_diffuseColor, tempcolor1);
			bgfx::setUniform(u_ambientColor, tempcolor2);
			if (m_showLightCounts)
			{
				meshSubmit(m_mesh, 1, m_program_light_dbg, mtx);
			}
			else
			{
				meshSubmit(m_mesh, 1, m_program_light, mtx);
			}

			// Advance to next frame. Rendering thread will be kicked to
			// process submitted rendering primitives.
			bgfx::frame();

			return true;
		}

		return false;
	}

	uint32_t m_width;
	uint32_t m_height;
	uint32_t m_debug;
	uint32_t m_reset;

	uint32_t m_tiles_x;
	uint32_t m_tiles_y;

	bool m_showLightCounts;
	int32_t m_scrollArea;

	int64_t m_timeOffset;
	Mesh* m_mesh;
	bgfx::ProgramHandle m_program_depthpass;
	bgfx::ProgramHandle m_program_compute;
	bgfx::ProgramHandle m_program_light;
	bgfx::ProgramHandle m_program_light_dbg;

	LightData m_lightData;

	bgfx::UniformHandle u_screenSize;
	bgfx::UniformHandle u_lightCount;
	bgfx::UniformHandle u_dispatchParams;
	bgfx::UniformHandle u_viewMat;
	bgfx::UniformHandle u_projectionMat;
	bgfx::UniformHandle u_projectionInvMat;
	bgfx::UniformHandle s_depthMap;

	bgfx::UniformHandle u_diffuseColor;
	bgfx::UniformHandle u_ambientColor;

	bgfx::TextureHandle m_fbtextures[2];
	bgfx::FrameBufferHandle m_fbh;

	entry::MouseState m_mouseState;
};

ENTRY_IMPLEMENT_MAIN(ExampleFPlus);
