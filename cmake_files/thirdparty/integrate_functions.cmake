macro(target_add_morphstore target)
    target_compile_options(${target} PUBLIC -DMSV_NO_SELFMANAGED_MEMORY)
    target_include_directories(${target} PUBLIC ${MorphStoreRoot}/include)
endmacro()
