/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_RUNTIME_LOCAL_IO_FILE_H
#define SRC_RUNTIME_LOCAL_IO_FILE_H

#include <stdio.h>
#include <stdlib.h>

struct File {
  FILE *identifier;
  unsigned long pos;
  long read;
  char *line;
  size_t line_len;
};

inline struct File *openMemFile(FILE *ident){
  struct File *f = (struct File *)malloc(sizeof(struct File));

  f->identifier = ident;
  f->pos = 0;

  f->line = NULL;
  f->line_len = 0;

  return f;
}

inline struct File *openFile(const char *filename) {
  struct File *f = (struct File *)malloc(sizeof(struct File));

  f->identifier = fopen(filename, "r");
  f->pos = 0;

  if (f->identifier == NULL)
    return NULL;

  f->line = NULL;
  f->line_len = 0;

  return f;
}

inline struct File *openFileForWrite(const char *filename) {
  struct File *f = (struct File *)malloc(sizeof(struct File));

  f->identifier = fopen(filename, "w+");
  f->pos = 0;
  
  if (f->identifier == NULL)
    return NULL;

  f->line = NULL;
  f->line_len = 0;

  return f;
}

inline void closeFile(File *f) {
  fclose(f->identifier);
  if (f->line) {
    free(f->line);
  }
}

inline ssize_t getFileLine(File *f) {
  ssize_t ret = getline(&f->line, &f->line_len, f->identifier);
  f->read = ret;
  f->pos += ret;

  return ret;
}

#endif
