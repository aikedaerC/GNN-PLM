int /* PRIVATE */
png_set_text_2(png_structp png_ptr, png_infop info_ptr, png_textp text_ptr, int num_text)
{
   int i;
   png_debug1(1, "in %s storage function", ((png_ptr == NULL || png_ptr->chunk_name[0] == '\0') ?
                                           "text" : (png_const_charp)png_ptr->chunk_name));
   if (png_ptr == NULL || info_ptr == NULL || num_text == 0)
       return(0);
    ...
}
