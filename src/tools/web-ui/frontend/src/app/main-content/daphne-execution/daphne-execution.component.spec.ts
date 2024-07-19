import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DaphneExecutionComponent } from './daphne-execution.component';

describe('DaphneExecutionComponent', () => {
  let component: DaphneExecutionComponent;
  let fixture: ComponentFixture<DaphneExecutionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ DaphneExecutionComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(DaphneExecutionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
