import { ComponentFixture, TestBed } from '@angular/core/testing';

import { IoComponent } from './io.component';

describe('IoComponent', () => {
  let component: IoComponent;
  let fixture: ComponentFixture<IoComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ IoComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(IoComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
