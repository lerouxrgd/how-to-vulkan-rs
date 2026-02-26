use std::ffi::{CString, c_char};
use std::{env, fs, mem, ptr, slice};

use anyhow::{anyhow, ensure};
use ash::vk;
use glam::{Vec2, Vec3};
use gpu_allocator::vulkan::{
    Allocation, AllocationCreateDesc, AllocationScheme, Allocator, AllocatorCreateDesc,
};
use gpu_allocator::{AllocatorDebugSettings, MemoryLocation};
use sdl3::sys::vulkan::{SDL_Vulkan_CreateSurface, SDL_Vulkan_GetPresentationSupport};

const MAX_FRAMES_IN_FLIGHT: usize = 2;

#[repr(C)]
struct Vertex {
    pos: Vec3,
    normal: Vec3,
    uv: Vec2,
}

#[repr(C)]
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

struct ShaderDataBuffer {
    allocation: Allocation,
    buffer: vk::Buffer,
    device_address: vk::DeviceAddress,
}

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

    // volkInitialize() equivalent
    let entry = unsafe { ash::Entry::load()? };

    // App info
    let app_name = c"How to Vulkan";
    let app_info = vk::ApplicationInfo::default()
        .application_name(app_name)
        .api_version(vk::API_VERSION_1_3);
    println!("ApplicationInfo created:");
    println!("  app name : {}", app_name.to_string_lossy());
    println!(
        "  api_version: {}.{}.{}",
        vk::api_version_major(app_info.api_version),
        vk::api_version_minor(app_info.api_version),
        vk::api_version_patch(app_info.api_version),
    );
    println!();

    // Ask SDL for the platform specific instance extensions
    let extensions = window.vulkan_instance_extensions()?;
    let extension_names: Vec<CString> = extensions
        .into_iter()
        .map(|e| CString::new(e).unwrap_or_default())
        .collect();
    let extension_ptrs: Vec<*const c_char> = extension_names.iter().map(|e| e.as_ptr()).collect();

    // Instance
    let instance_ci = vk::InstanceCreateInfo::default()
        .application_info(&app_info)
        .enabled_extension_names(&extension_ptrs);
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
    println!();

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
    let (w_width, w_height) = window.size();
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
    let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_ci, None)? };
    let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
    let swapchain_image_views = swapchain_images
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
    let depth_image = unsafe { device.create_image(&depth_image_ci, None)? };
    let requirements = unsafe { device.get_image_memory_requirements(depth_image) };
    let depth_image_allocation = allocator.allocate(&AllocationCreateDesc {
        name: "depth_image",
        requirements,
        location: MemoryLocation::GpuOnly, // here equivalent to VMA_MEMORY_USAGE_AUTO
        linear: false, // VK_IMAGE_TILING_OPTIMAL, the GPU arranges texels in a tiled layout for better cache performance
        allocation_scheme: AllocationScheme::DedicatedImage(depth_image), // VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT
    })?;
    unsafe {
        device.bind_image_memory(
            depth_image,
            depth_image_allocation.memory(),
            depth_image_allocation.offset(),
        )?
    };
    let depth_image_view = unsafe {
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
    let (models, _materials) = tobj::load_obj("assets/suzanne.obj", &tobj::LoadOptions::default())?;
    let mesh = &models[0].mesh;
    let mut vertices: Vec<Vertex> = Vec::new();
    let mut indices: Vec<u16> = Vec::new();
    for i in 0..mesh.indices.len() {
        let vi = mesh.indices[i] as usize;
        let ni = mesh.normal_indices[i] as usize;
        let ti = mesh.texcoord_indices[i] as usize;
        let v = Vertex {
            pos: Vec3::new(
                mesh.positions[vi * 3],
                -mesh.positions[vi * 3 + 1],
                mesh.positions[vi * 3 + 2],
            ),
            normal: Vec3::new(
                mesh.normals[ni * 3],
                -mesh.normals[ni * 3 + 1],
                mesh.normals[ni * 3 + 2],
            ),
            uv: Vec2::new(mesh.texcoords[ti * 2], 1.0 - mesh.texcoords[ti * 2 + 1]),
        };
        indices.push(indices.len() as u16);
        vertices.push(v);
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
    let mapped_ptr = match v_buffer_allocation.mapped_ptr() {
        Some(ptr) => ptr.as_ptr() as *mut u8,
        None => anyhow::bail!(
            "ReBAR/SAM not available — vertex buffer not in host-visible device-local memory"
        ),
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
    for i in 0..3 {
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
        textures.push(Texture {
            allocation,
            image,
            view,
            sampler: vk::Sampler::null(),
        });

        // Upload
    }

    Ok(())
}
