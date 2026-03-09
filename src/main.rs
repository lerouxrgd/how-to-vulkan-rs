use std::ffi::{CString, c_char};
use std::{env, fs, mem, ptr, slice};

use anyhow::{anyhow, bail, ensure};
use ash::vk;
use bytemuck::{bytes_of, cast_slice};
use glam::{Vec2, Vec3};
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::{AllocatorDebugSettings, MemoryLocation};
use sdl3::sys::vulkan::{SDL_Vulkan_CreateSurface, SDL_Vulkan_GetPresentationSupport};
use slang::{CompileTarget, Downcast};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[repr(C)]
struct Vertex {
    pos: Vec3,
    normal: Vec3,
    uv: Vec2,
}

#[repr(C)]
#[derive(Clone)]
struct ShaderData {
    projection: glam::Mat4,
    view: glam::Mat4,
    model: [glam::Mat4; 3],
    light_pos: glam::Vec4,
    selected: u32,
}

impl Default for ShaderData {
    fn default() -> Self {
        Self {
            projection: glam::Mat4::IDENTITY,
            view: glam::Mat4::IDENTITY,
            model: [glam::Mat4::IDENTITY; 3],
            light_pos: glam::Vec4::new(0.0, -10.0, 10.0, 0.0),
            selected: 1,
        }
    }
}

#[allow(dead_code)]
struct ShaderDataBuffer {
    allocation: Allocation,
    buffer: vk::Buffer,
    device_address: vk::DeviceAddress,
}

#[allow(dead_code)]
struct Texture {
    allocation: gpu_allocator::vulkan::Allocation,
    image: vk::Image,
    view: vk::ImageView,
    sampler: vk::Sampler,
}

fn main() -> anyhow::Result<()> {
    // SDL_Init(SDL_INIT_VIDEO)
    let sdl_context = sdl3::init()?;
    let video = sdl_context.video()?;

    // Create window
    let window = video
        .window("How to Vulkan", 1080, 720)
        .vulkan()
        .resizable()
        .build()?;

    // Equivalent to volkInitialize()
    let entry = unsafe { ash::Entry::load()? };

    // App info
    let app_name = c"How to Vulkan";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .api_version(vk::API_VERSION_1_3);

    // Ask SDL for the platform specific instance extensions
    let extensions = window.vulkan_instance_extensions()?;
    let extension_names: Vec<CString> = extensions
        .into_iter()
        .map(|e| CString::new(e).unwrap_or_default())
        .collect();
    let extension_ptrs: Vec<*const c_char> = extension_names.iter().map(|e| e.as_ptr()).collect();

    // Validation layers
    let layer_names = [c"VK_LAYER_KHRONOS_validation".as_ptr()];

    // Instance
    let instance_ci = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_ptrs)
        .enabled_layer_names(&layer_names);
    let instance = unsafe { entry.create_instance(&instance_ci, None)? };

    // Devices
    let devices = unsafe { instance.enumerate_physical_devices()? };
    // Device selection via command line argument (default: 0)
    let device_index: usize = env::args().nth(1).and_then(|a| a.parse().ok()).unwrap_or(0);
    ensure!(device_index < devices.len(), "Device index out of range");

    // Physical device
    let physical_device = devices[device_index];
    let mut device_properties = vk::PhysicalDeviceProperties2::default();
    unsafe { instance.get_physical_device_properties2(physical_device, &mut device_properties) };
    let device_name = device_properties
        .properties
        .device_name_as_c_str()?
        .to_string_lossy();
    println!("Selected device: {device_name}");

    // Find a queue family that supports graphics and presentation
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
    let queue_family = queue_families
        .iter()
        .enumerate()
        .find(|(_, qf)| qf.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|(i, _)| i as u32)
        .ok_or_else(|| anyhow!("No graphics queue family found"))?;
    let presentation_support = unsafe {
        SDL_Vulkan_GetPresentationSupport(instance.handle(), physical_device, queue_family)
    };
    ensure!(
        presentation_support,
        "Queue family does not support presentation"
    );
    let queue_priorities = [1.0]; // implies queue_count == 1
    let queue_ci = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family)
        .queue_priorities(&queue_priorities);

    // Use features from previous Vulkan versions
    let mut enabled_vk12_features = vk::PhysicalDeviceVulkan12Features::default()
        .descriptor_indexing(true)
        .shader_sampled_image_array_non_uniform_indexing(true) // related to descriptor indexing
        .descriptor_binding_variable_descriptor_count(true) // related to descriptor indexing
        .runtime_descriptor_array(true) // related to descriptor indexing
        .buffer_device_address(true);
    let mut enabled_vk13_features = vk::PhysicalDeviceVulkan13Features::default()
        .synchronization2(true)
        .dynamic_rendering(true);
    let enabled_vk10_features = vk::PhysicalDeviceFeatures::default().sampler_anisotropy(true); // better texture filtering
    // Require swapchain device extension
    let device_extensions = [ash::khr::swapchain::NAME.as_ptr()];

    // Logical device
    let device_ci = vk::DeviceCreateInfo::default()
        .queue_create_infos(slice::from_ref(&queue_ci))
        .enabled_extension_names(&device_extensions)
        .enabled_features(&enabled_vk10_features)
        .push_next(&mut enabled_vk13_features)
        .push_next(&mut enabled_vk12_features);
    let device = unsafe { instance.create_device(physical_device, &device_ci, None)? };
    let queue = unsafe { device.get_device_queue(queue_family, 0) }; // queue index 0 in queue_priorities

    // GPU memory allocator
    let mut allocator = Allocator::new(&AllocatorCreateDesc {
        instance: instance.clone(),
        device: device.clone(),
        physical_device,
        debug_settings: AllocatorDebugSettings::default(),
        buffer_device_address: true,
        allocation_sizes: Default::default(),
    })?;

    // Surface
    let mut surface = vk::SurfaceKHR::null();
    unsafe {
        ensure!(
            SDL_Vulkan_CreateSurface(
                window.raw(),
                instance.handle(),
                std::ptr::null(),
                &mut surface,
            ),
            "SDL_Vulkan_CreateSurface failed"
        );
    }
    let surface_loader = ash::khr::surface::Instance::new(&entry, &instance);
    let surface_caps = unsafe {
        surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?
    };
    let (mut w_width, mut w_height) = window.size();
    let extent = if surface_caps.current_extent.width != u32::MAX {
        surface_caps.current_extent
    } else {
        // Here u32::MAX meant "the extent is determined by the swapchain", so fall back to the window size
        vk::Extent2D {
            width: w_width.clamp(
                surface_caps.min_image_extent.width,
                surface_caps.max_image_extent.width,
            ),
            height: w_height.clamp(
                surface_caps.min_image_extent.height,
                surface_caps.max_image_extent.height,
            ),
        }
    };

    // Swap chain
    let image_format = vk::Format::B8G8R8A8_SRGB;
    let swapchain_ci = vk::SwapchainCreateInfoKHR::default()
        .surface(surface)
        .min_image_count(surface_caps.min_image_count)
        .image_format(image_format)
        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
        .image_extent(extent)
        .image_array_layers(1)
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(vk::PresentModeKHR::FIFO);
    let swapchain_loader = ash::khr::swapchain::Device::new(&instance, &device);
    let mut swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_ci, None)? };
    let mut swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
    let mut swapchain_image_views = swapchain_images
        .iter()
        .map(|&image| {
            let view_ci = vk::ImageViewCreateInfo::default()
                .image(image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(image_format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            unsafe { device.create_image_view(&view_ci, None) }
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Depth attachment
    let depth_format_candidates = [
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];
    let depth_format = depth_format_candidates
        .iter()
        .copied()
        .find(|&format| {
            let mut format_properties = vk::FormatProperties2::default();
            unsafe {
                instance.get_physical_device_format_properties2(
                    physical_device,
                    format,
                    &mut format_properties,
                )
            };
            format_properties
                .format_properties
                .optimal_tiling_features
                .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
        })
        .ok_or_else(|| anyhow::anyhow!("No suitable depth format found"))?;
    let depth_image_ci = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(depth_format)
        .extent(vk::Extent3D {
            width: w_width,
            height: w_height,
            depth: 1,
        })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1) // no multi sampling
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let mut depth_image = unsafe { device.create_image(&depth_image_ci, None)? };
    let requirements = unsafe { device.get_image_memory_requirements(depth_image) };
    let mut depth_image_allocation = allocator.allocate(&AllocationCreateDesc {
        name: "depth_image",
        requirements,
        location: MemoryLocation::GpuOnly, // here equivalent to VMA_MEMORY_USAGE_AUTO
        linear: false,
        allocation_scheme: AllocationScheme::DedicatedImage(depth_image), // VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    })?;
    unsafe {
        device.bind_image_memory(
            depth_image,
            depth_image_allocation.memory(),
            depth_image_allocation.offset(),
        )?
    };
    let mut depth_image_view = unsafe {
        device.create_image_view(
            &vk::ImageViewCreateInfo::default()
                .image(depth_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(depth_format)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                ),
            None,
        )?
    };

    // Mesh data
    let (models, _materials) = tobj::load_obj(
        "assets/suzanne.obj",
        &tobj::LoadOptions {
            triangulate: true,
            single_index: true,
            ..Default::default()
        },
    )?;
    let mesh = &models[0].mesh;
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    for i in 0..mesh.indices.len() {
        let vi = mesh.indices[i] as usize;
        vertices.push(Vertex {
            pos: Vec3::new(
                mesh.positions[vi * 3],
                -mesh.positions[vi * 3 + 1],
                mesh.positions[vi * 3 + 2],
            ),
            normal: Vec3::new(
                mesh.normals[vi * 3],
                -mesh.normals[vi * 3 + 1],
                mesh.normals[vi * 3 + 2],
            ),
            uv: Vec2::new(mesh.texcoords[vi * 2], 1.0 - mesh.texcoords[vi * 2 + 1]),
        });
        indices.push(i as u16);
    }
    let v_buf_size = mem::size_of::<Vertex>() * vertices.len();
    let i_buf_size = mem::size_of::<u16>() * indices.len();
    let buffer_ci = vk::BufferCreateInfo::default()
        .size((v_buf_size + i_buf_size) as u64) // vertices and indices will both be put into this buffer
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::INDEX_BUFFER);
    let v_buffer = unsafe { device.create_buffer(&buffer_ci, None)? };
    let requirements = unsafe { device.get_buffer_memory_requirements(v_buffer) };
    let v_buffer_allocation = allocator.allocate(&AllocationCreateDesc {
        name: "vertex_index_buffer",
        requirements,
        location: MemoryLocation::CpuToGpu,
        linear: true, // data laid out sequentially
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;
    unsafe {
        device.bind_buffer_memory(
            v_buffer,
            v_buffer_allocation.memory(),
            v_buffer_allocation.offset(),
        )?
    };
    let mapped_ptr = match v_buffer_allocation.mapped_ptr() {
        Some(ptr) => ptr.as_ptr() as *mut u8,
        None => {
            bail!("ReBAR/SAM not available, vertex buffer not in host-visible device-local memory")
        }
    };
    unsafe {
        // Cast `as *const u8` to treat the memory as raw bytes and allow pointer arithmetic with .add(v_buf_size)
        ptr::copy_nonoverlapping(vertices.as_ptr() as *const u8, mapped_ptr, v_buf_size);
        ptr::copy_nonoverlapping(
            indices.as_ptr() as *const u8,
            mapped_ptr.add(v_buf_size),
            i_buf_size,
        );
    }

    // Shader data buffers
    let mut shader_data_buffers: Vec<ShaderDataBuffer> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        let u_buffer_ci = vk::BufferCreateInfo::default()
            .size(mem::size_of::<ShaderData>() as u64)
            .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS);
        let buffer = unsafe { device.create_buffer(&u_buffer_ci, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "shader_data_buffer",
            requirements,
            location: MemoryLocation::CpuToGpu,
            linear: true, // data laid out sequentially
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };
        let device_address = unsafe {
            device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        shader_data_buffers.push(ShaderDataBuffer {
            allocation,
            buffer,
            device_address,
        });
    }

    // Synchronization objects
    let semaphore_ci = vk::SemaphoreCreateInfo::default();
    let fence_ci = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
    let mut fences: Vec<vk::Fence> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
    let mut present_semaphores: Vec<vk::Semaphore> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        fences.push(unsafe { device.create_fence(&fence_ci, None)? });
        present_semaphores.push(unsafe { device.create_semaphore(&semaphore_ci, None)? });
    }
    let render_semaphores = swapchain_images
        .iter()
        .map(|_| unsafe { device.create_semaphore(&semaphore_ci, None) })
        .collect::<Result<Vec<_>, _>>()?;

    // Command buffers
    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family),
            None,
        )?
    };
    let command_buffers = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .command_buffer_count(MAX_FRAMES_IN_FLIGHT as u32),
        )?
    };

    // Texture images
    let mut textures: Vec<Texture> = Vec::with_capacity(3);
    let mut texture_descriptors: Vec<vk::DescriptorImageInfo> = Vec::with_capacity(3);
    for i in 0..3 {
        // Read KTX2 texture
        let ktx_data = fs::read(format!("assets/suzanne{}.ktx2", i))?;
        let reader =
            ktx2::Reader::new(&ktx_data).map_err(|e| anyhow!("Failed to read KTX2: {:?}", e))?;
        let header = reader.header();
        let format = vk::Format::from_raw(
            header
                .format
                .ok_or_else(|| anyhow!("KTX2 has no format"))?
                .value() as i32,
        );

        // Texture image and view
        let tex_image_ci = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: header.pixel_width,
                height: header.pixel_height,
                depth: 1,
            })
            .mip_levels(header.level_count)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = unsafe { device.create_image(&tex_image_ci, None)? };
        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let allocation = allocator.allocate(&AllocationCreateDesc {
            name: "texture_image",
            requirements,
            location: MemoryLocation::GpuOnly,
            linear: false,
            allocation_scheme: AllocationScheme::DedicatedImage(image),
        })?;
        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset())? };
        let view = unsafe {
            device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(header.level_count)
                            .layer_count(1),
                    ),
                None,
            )?
        };

        // Upload texture (through texture staging buffer)
        let levels: Vec<&[u8]> = reader.levels().map(|l| l.data).collect();
        let data_size: usize = levels.iter().map(|l| l.len()).sum();
        let mut ktx_data_flat: Vec<u8> = Vec::with_capacity(data_size);
        for level in &levels {
            ktx_data_flat.extend_from_slice(level);
        }
        let img_src_buffer_ci = vk::BufferCreateInfo::default()
            .size(data_size as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        let img_src_buffer = unsafe { device.create_buffer(&img_src_buffer_ci, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(img_src_buffer) };
        let img_src_allocation = allocator.allocate(&AllocationCreateDesc {
            name: "texture_staging_buffer",
            requirements,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        })?;
        unsafe {
            device.bind_buffer_memory(
                img_src_buffer,
                img_src_allocation.memory(),
                img_src_allocation.offset(),
            )?
        };
        let mapped_ptr = match img_src_allocation.mapped_ptr() {
            Some(ptr) => ptr.as_ptr() as *mut u8,
            None => bail!("Staging buffer not host-visible"),
        };
        unsafe {
            ptr::copy_nonoverlapping(ktx_data_flat.as_ptr(), mapped_ptr, data_size);
        }
        // A fence and command buffer just for this upload
        let fence_one_time = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let cb_one_time = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .command_buffer_count(1),
            )?
        }[0];
        unsafe {
            device.begin_command_buffer(
                cb_one_time,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?
        };
        // Barrier: UNDEFINED -> TRANSFER_DST_OPTIMAL
        let barrier_transfer = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(header.level_count)
                    .layer_count(1),
            );
        unsafe {
            device.cmd_pipeline_barrier2(
                cb_one_time,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(slice::from_ref(&barrier_transfer)),
            )
        };
        // Copy regions, one per mip level
        let mut offset = 0;
        let copy_regions: Vec<vk::BufferImageCopy> = levels
            .iter()
            .enumerate()
            .map(|(j, level)| {
                let region = vk::BufferImageCopy::default()
                    .buffer_offset(offset as u64)
                    .image_subresource(
                        vk::ImageSubresourceLayers::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .mip_level(j as u32)
                            .layer_count(1),
                    )
                    .image_extent(vk::Extent3D {
                        // >> j halves the dimensions at each level
                        width: header.pixel_width >> j,
                        height: header.pixel_height >> j,
                        depth: 1,
                    });
                offset += level.len();
                region
            })
            .collect();
        unsafe {
            device.cmd_copy_buffer_to_image(
                cb_one_time,
                img_src_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &copy_regions,
            )
        };
        // Barrier: TRANSFER_DST_OPTIMAL -> READ_ONLY_OPTIMAL
        let barrier_read = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::FRAGMENT_SHADER)
            .dst_access_mask(vk::AccessFlags2::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)
            .image(image)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(header.level_count)
                    .layer_count(1),
            );
        unsafe {
            device.cmd_pipeline_barrier2(
                cb_one_time,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(slice::from_ref(&barrier_read)),
            )
        };
        // Submit, wait, cleanup
        unsafe { device.end_command_buffer(cb_one_time)? };
        unsafe {
            device.queue_submit(
                queue,
                &[vk::SubmitInfo::default().command_buffers(slice::from_ref(&cb_one_time))],
                fence_one_time,
            )?
        };
        unsafe { device.wait_for_fences(&[fence_one_time], true, u64::MAX)? };
        unsafe { device.destroy_fence(fence_one_time, None) };
        allocator.free(img_src_allocation)?;
        unsafe { device.destroy_buffer(img_src_buffer, None) };

        // Sampler
        let sampler = unsafe {
            device.create_sampler(
                &vk::SamplerCreateInfo::default()
                    .mag_filter(vk::Filter::LINEAR)
                    .min_filter(vk::Filter::LINEAR)
                    .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
                    .anisotropy_enable(true)
                    .max_anisotropy(8.0)
                    .max_lod(header.level_count as f32),
                None,
            )?
        };
        textures.push(Texture {
            allocation,
            image,
            view,
            sampler,
        });
        texture_descriptors.push(
            vk::DescriptorImageInfo::default()
                .sampler(sampler)
                .image_view(view)
                .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL),
        );
    }

    // Descriptor (indexing)
    let desc_variable_flag = vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;
    let mut desc_binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
        .binding_flags(slice::from_ref(&desc_variable_flag));
    let desc_layout_binding_tex = vk::DescriptorSetLayoutBinding::default()
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(textures.len() as u32)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT);
    let descriptor_set_layout_tex = unsafe {
        device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default()
                .bindings(slice::from_ref(&desc_layout_binding_tex))
                .push_next(&mut desc_binding_flags),
            None,
        )?
    };
    let pool_size = vk::DescriptorPoolSize::default()
        .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(textures.len() as u32);
    let descriptor_pool = unsafe {
        device.create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(1) // How many descriptor sets (one of dynamic size)
                .pool_sizes(slice::from_ref(&pool_size)),
            None,
        )?
    };
    let variable_desc_count = textures.len() as u32;
    let mut variable_desc_count_ai =
        vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(slice::from_ref(&variable_desc_count));
    let descriptor_set_tex = unsafe {
        device.allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(descriptor_pool)
                .set_layouts(slice::from_ref(&descriptor_set_layout_tex))
                .push_next(&mut variable_desc_count_ai),
        )?
    }[0];
    let write_desc_set = vk::WriteDescriptorSet::default()
        .dst_set(descriptor_set_tex)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .image_info(&texture_descriptors);
    unsafe { device.update_descriptor_sets(slice::from_ref(&write_desc_set), &[]) };

    // Initialize Slang shader compiler
    let global_session = slang::GlobalSession::new()
        .ok_or_else(|| anyhow!("Failed to create Slang global session"))?;
    let target_desc = slang::TargetDesc::default()
        .format(CompileTarget::Spirv)
        .profile(global_session.find_profile("spirv_1_4"));
    let session_options = slang::CompilerOptions::default()
        .emit_spirv_directly(true)
        .matrix_layout_column(true);
    let session_desc = slang::SessionDesc::default()
        .targets(slice::from_ref(&target_desc))
        .options(&session_options);
    let session = global_session
        .create_session(&session_desc)
        .ok_or_else(|| anyhow!("Failed to create Slang session"))?;

    // Load shader
    let module = session.load_module("assets/shader.slang")?;
    let vert_entry_point = module
        .entry_point_by_index(0) // vertex "main"
        .ok_or_else(|| anyhow!("Failed to find vertex entry point"))?;
    let frag_entry_point = module
        .entry_point_by_index(1) // fragment "main"
        .ok_or_else(|| anyhow!("Failed to find fragment entry point"))?;
    let program = session.create_composite_component_type(&[
        module.downcast().clone(),
        vert_entry_point.downcast().clone(),
        frag_entry_point.downcast().clone(),
    ])?;
    let linked_program = program.link()?; // resolve all cross-module references, optimizations
    let vert_spirv = linked_program.entry_point_code(0, 0)?; // entry_point 0, target 0 (spirv_1_4)
    let frag_spirv = linked_program.entry_point_code(1, 0)?; // entry_point 1, target 0 (spirv_1_4)
    let vert_shader_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(cast_slice(vert_spirv.as_slice())),
            None,
        )?
    };
    let frag_shader_module = unsafe {
        device.create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(cast_slice(frag_spirv.as_slice())),
            None,
        )?
    };

    // Pipeline
    let push_constant_range = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .size(mem::size_of::<vk::DeviceAddress>() as u32);
    let pipeline_layout = unsafe {
        device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                // define the interface to the shader resources
                .set_layouts(slice::from_ref(&descriptor_set_layout_tex))
                // pass the VkDeviceAddress of the ShaderData buffer to the vertex shader
                .push_constant_ranges(slice::from_ref(&push_constant_range)),
            None,
        )?
    };
    let vertex_binding = vk::VertexInputBindingDescription::default()
        .binding(0)
        .stride(mem::size_of::<Vertex>() as u32)
        .input_rate(vk::VertexInputRate::VERTEX);
    let vertex_attributes = [
        vk::VertexInputAttributeDescription::default()
            .location(0)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT), // 3×f32 for `pos`
        vk::VertexInputAttributeDescription::default()
            .location(1)
            .binding(0)
            .format(vk::Format::R32G32B32_SFLOAT) // 3×f32 for `normal`
            .offset(mem::offset_of!(Vertex, normal) as u32),
        vk::VertexInputAttributeDescription::default()
            .location(2)
            .binding(0)
            .format(vk::Format::R32G32_SFLOAT) // 2×f32 for `uv`
            .offset(mem::offset_of!(Vertex, uv) as u32),
    ];
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(slice::from_ref(&vertex_binding))
        .vertex_attribute_descriptions(&vertex_attributes);
    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let shader_stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(vert_shader_module)
            .name(c"main"),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(frag_shader_module)
            .name(c"main"),
    ];
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default().line_width(1.0);
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1); // no MSAA (as in swapchain and depth image)
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
        .attachments(slice::from_ref(&blend_attachment));
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
    let mut rendering_ci = vk::PipelineRenderingCreateInfo::default() // VK_KHR_dynamic_rendering
        .color_attachment_formats(slice::from_ref(&image_format))
        .depth_attachment_format(depth_format);
    let pipeline = unsafe {
        device
            .create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[vk::GraphicsPipelineCreateInfo::default()
                    .push_next(&mut rendering_ci)
                    .stages(&shader_stages)
                    .vertex_input_state(&vertex_input_state)
                    .input_assembly_state(&input_assembly_state)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterization_state)
                    .multisample_state(&multisample_state)
                    .depth_stencil_state(&depth_stencil_state)
                    .color_blend_state(&color_blend_state)
                    .dynamic_state(&dynamic_state)
                    .layout(pipeline_layout)],
                None,
            )
            .map_err(|(_, e)| e)?[0]
    };

    // Render loop
    let mut cam_pos = glam::Vec3::new(0.0, 0.0, -6.0);
    let mut object_rotations = [glam::Vec3::ZERO; 3];
    let mut event_pump = sdl_context.event_pump()?;
    let mut update_swapchain = false;
    let mut shader_data = ShaderData::default();
    let mut last_time = sdl3::timer::ticks();
    let mut current_frame = 0;
    let mut quit = false;
    while !quit {
        // Wait on fence
        unsafe {
            device.wait_for_fences(&[fences[current_frame]], true, u64::MAX)?;
            device.reset_fences(&[fences[current_frame]])?;
        }

        // Acquire next image
        let image_index = unsafe {
            swapchain_loader
                .acquire_next_image(
                    swapchain,
                    u64::MAX,
                    present_semaphores[current_frame],
                    vk::Fence::null(),
                )?
                .0 as usize
        };

        // Update shader data
        let aspect_ratio = w_width as f32 / w_height as f32;
        shader_data.projection =
            glam::Mat4::perspective_rh(45.0f32.to_radians(), aspect_ratio, 0.1, 32.0);
        shader_data.view = glam::Mat4::from_translation(cam_pos);
        for (i, object_rotation) in object_rotations.iter().enumerate() {
            let instance_pos = glam::Vec3::new((i as f32 - 1.0) * 3.0, 0.0, 0.0);
            shader_data.model[i] = glam::Mat4::from_translation(instance_pos)
                * glam::Mat4::from_euler(
                    glam::EulerRot::XYZ,
                    object_rotation.x,
                    object_rotation.y,
                    object_rotation.z,
                );
        }
        let mapped_ptr = shader_data_buffers[current_frame]
            .allocation
            .mapped_ptr()
            .ok_or_else(|| anyhow!("Shader data buffer not mapped"))?
            .as_ptr() as *mut ShaderData;
        unsafe { ptr::write(mapped_ptr, shader_data.clone()) };

        // Record command buffer
        let cb = command_buffers[current_frame];
        unsafe {
            device.reset_command_buffer(cb, vk::CommandBufferResetFlags::empty())?;
            device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    // affects how lifecycle moves to invalid state after execution and
                    // can be used as an optimization hint by drivers
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }
        let output_barriers = [
            vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::empty())
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(
                    // means the render pass will both read and write the color attachment
                    vk::AccessFlags2::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                )
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                .image(swapchain_images[image_index])
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                ),
            vk::ImageMemoryBarrier2::default()
                // previous frame may have written depth in the late fragment test stage
                .src_stage_mask(vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
                .src_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                // this frame needs to write depth starting from the early fragment test stage
                .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS)
                .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
                .image(depth_image)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL)
                        .level_count(1)
                        .layer_count(1),
                ),
        ];
        unsafe {
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default().image_memory_barriers(&output_barriers),
            )
        };
        let color_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(swapchain_image_views[image_index])
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR) // clear value at the start
            .store_op(vk::AttachmentStoreOp::STORE) // keep after rendering so it can be presented
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0], // black
                },
            });
        let depth_attachment_info = vk::RenderingAttachmentInfo::default()
            .image_view(depth_image_view)
            .image_layout(vk::ImageLayout::ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE) // don't save the depth buffer after rendering
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0, // maximum depth, so everything renders in front of it
                    stencil: 0,
                },
            });
        unsafe {
            device.cmd_begin_rendering(
                cb,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: vk::Extent2D {
                            width: w_width,
                            height: w_height,
                        },
                    })
                    .layer_count(1)
                    .color_attachments(slice::from_ref(&color_attachment_info))
                    .depth_attachment(&depth_attachment_info),
            );
            device.cmd_set_viewport(
                cb,
                0,
                &[vk::Viewport {
                    x: 0.0,
                    y: 0.0,
                    width: w_width as f32,
                    height: w_height as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                }],
            );
            device.cmd_set_scissor(
                cb,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D::default(),
                    extent: vk::Extent2D {
                        width: w_width,
                        height: w_height,
                    },
                }],
            );
            device.cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_layout,
                0,
                &[descriptor_set_tex],
                &[],
            );
            device.cmd_bind_vertex_buffers(cb, 0, &[v_buffer], &[0]);
            device.cmd_bind_index_buffer(cb, v_buffer, v_buf_size as u64, vk::IndexType::UINT16);
            device.cmd_push_constants(
                cb,
                pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                bytes_of(&shader_data_buffers[current_frame].device_address),
            );
            device.cmd_draw_indexed(cb, indices.len() as u32, 3, 0, 0, 0);
            device.cmd_end_rendering(cb);
        }
        let barrier_present = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::empty())
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR) // ready to be presented to the screen
            .image(swapchain_images[image_index])
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        unsafe {
            device.cmd_pipeline_barrier2(
                cb,
                &vk::DependencyInfo::default()
                    .image_memory_barriers(slice::from_ref(&barrier_present)),
            );
            device.end_command_buffer(cb)?;
        }

        // Submit command buffer
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        unsafe {
            device.queue_submit(
                queue,
                &[vk::SubmitInfo::default()
                    // don't start color output until the swapchain image is ready to be
                    // written to (signaled by acquire_next_image)
                    .wait_semaphores(slice::from_ref(&present_semaphores[current_frame]))
                    .wait_dst_stage_mask(&wait_stages)
                    .command_buffers(slice::from_ref(&cb))
                    // tell the presenter that rendering is done and the image can be presented
                    .signal_semaphores(slice::from_ref(&render_semaphores[image_index]))],
                fences[current_frame],
            )?;
        }
        current_frame = (current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

        // Present image
        let present_result = unsafe {
            swapchain_loader.queue_present(
                queue,
                &vk::PresentInfoKHR::default()
                    .wait_semaphores(slice::from_ref(&render_semaphores[image_index]))
                    .swapchains(slice::from_ref(&swapchain))
                    .image_indices(slice::from_ref(&(image_index as u32))),
            )
        };
        match present_result {
            Ok(_) => {}
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => update_swapchain = true,
            Err(e) => bail!(e),
        }

        // Poll events
        let elapsed_time = (sdl3::timer::ticks() - last_time) as f32 / 1000.0;
        last_time = sdl3::timer::ticks();
        for event in event_pump.poll_iter() {
            match event {
                // Exit loop if the application is about to close
                sdl3::event::Event::Quit { .. } => {
                    quit = true;
                    break;
                }

                // Rotate the selected object with mouse drag
                sdl3::event::Event::MouseMotion {
                    xrel,
                    yrel,
                    mousestate,
                    ..
                } => {
                    if mousestate.left() {
                        object_rotations[shader_data.selected as usize].x -= yrel * elapsed_time;
                        object_rotations[shader_data.selected as usize].y += xrel * elapsed_time;
                    }
                }

                // Zooming with the mouse wheel
                sdl3::event::Event::MouseWheel { y, .. } => {
                    cam_pos.z += y * elapsed_time * 10.0;
                }

                // Select active model instance
                sdl3::event::Event::KeyDown {
                    keycode: Some(key), ..
                } => match key {
                    sdl3::keyboard::Keycode::Plus | sdl3::keyboard::Keycode::KpPlus => {
                        shader_data.selected = if shader_data.selected < 2 {
                            shader_data.selected + 1
                        } else {
                            0
                        };
                    }
                    sdl3::keyboard::Keycode::Minus | sdl3::keyboard::Keycode::KpMinus => {
                        shader_data.selected = if shader_data.selected > 0 {
                            shader_data.selected - 1
                        } else {
                            2
                        };
                    }
                    _ => {}
                },

                // Window resize
                sdl3::event::Event::Window {
                    win_event: sdl3::event::WindowEvent::Resized(..),
                    ..
                } => {
                    update_swapchain = true;
                }
                _ => {}
            }
        }

        if update_swapchain {
            update_swapchain = false;
            let (new_width, new_height) = window.size();
            w_width = new_width;
            w_height = new_height;
            unsafe { device.device_wait_idle()? };
            let old_swapchain = swapchain;
            let new_swapchain = unsafe {
                swapchain_loader.create_swapchain(
                    &swapchain_ci
                        .old_swapchain(old_swapchain)
                        .image_extent(vk::Extent2D {
                            width: w_width,
                            height: w_height,
                        }),
                    None,
                )?
            };
            // Destroy old image views
            for &view in &swapchain_image_views {
                unsafe { device.destroy_image_view(view, None) };
            }
            // Get new swapchain images and recreate views
            swapchain = new_swapchain;
            swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
            swapchain_image_views = swapchain_images
                .iter()
                .map(|&image| unsafe {
                    device.create_image_view(
                        &vk::ImageViewCreateInfo::default()
                            .image(image)
                            .view_type(vk::ImageViewType::TYPE_2D)
                            .format(image_format)
                            .subresource_range(
                                vk::ImageSubresourceRange::default()
                                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                                    .level_count(1)
                                    .layer_count(1),
                            ),
                        None,
                    )
                })
                .collect::<Result<Vec<_>, _>>()?;
            // Destroy old swapchain
            unsafe { swapchain_loader.destroy_swapchain(old_swapchain, None) };
            // Recreate depth image
            allocator.free(depth_image_allocation)?;
            unsafe {
                device.destroy_image_view(depth_image_view, None);
                device.destroy_image(depth_image, None);
            }
            let new_depth_image = unsafe {
                device.create_image(
                    &depth_image_ci.extent(vk::Extent3D {
                        width: w_width,
                        height: w_height,
                        depth: 1,
                    }),
                    None,
                )?
            };
            let requirements = unsafe { device.get_image_memory_requirements(new_depth_image) };
            let new_depth_allocation = allocator.allocate(&AllocationCreateDesc {
                name: "depth_image",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::DedicatedImage(new_depth_image),
            })?;
            unsafe {
                device.bind_image_memory(
                    new_depth_image,
                    new_depth_allocation.memory(),
                    new_depth_allocation.offset(),
                )?
            };
            depth_image = new_depth_image;
            depth_image_allocation = new_depth_allocation;
            depth_image_view = unsafe {
                device.create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(depth_image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(depth_format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .level_count(1)
                                .layer_count(1),
                        ),
                    None,
                )?
            };
        }
    }

    // Tear down
    unsafe { device.device_wait_idle()? };
    for i in 0..MAX_FRAMES_IN_FLIGHT {
        unsafe {
            device.destroy_fence(fences[i], None);
            device.destroy_semaphore(present_semaphores[i], None);
        }
    }
    for sdb in shader_data_buffers.drain(..) {
        allocator.free(sdb.allocation)?;
        unsafe { device.destroy_buffer(sdb.buffer, None) };
    }
    for &semaphore in &render_semaphores {
        unsafe { device.destroy_semaphore(semaphore, None) };
    }
    allocator.free(depth_image_allocation)?;
    unsafe {
        device.destroy_image_view(depth_image_view, None);
        device.destroy_image(depth_image, None);
    }
    for &view in &swapchain_image_views {
        unsafe { device.destroy_image_view(view, None) };
    }
    allocator.free(v_buffer_allocation)?;
    unsafe { device.destroy_buffer(v_buffer, None) };
    for texture in textures {
        unsafe {
            device.destroy_image_view(texture.view, None);
            device.destroy_sampler(texture.sampler, None);
            device.destroy_image(texture.image, None);
        }
        allocator.free(texture.allocation)?;
    }
    unsafe {
        device.destroy_descriptor_set_layout(descriptor_set_layout_tex, None);
        device.destroy_descriptor_pool(descriptor_pool, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_pipeline(pipeline, None);
        swapchain_loader.destroy_swapchain(swapchain, None);
        surface_loader.destroy_surface(surface, None);
        device.destroy_command_pool(command_pool, None);
        device.destroy_shader_module(vert_shader_module, None);
        device.destroy_shader_module(frag_shader_module, None);
    }
    drop(allocator);
    unsafe {
        device.destroy_device(None);
        instance.destroy_instance(None);
    }

    Ok(())
}
